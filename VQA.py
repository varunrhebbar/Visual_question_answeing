import os, sys
import torch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Cleaner demos : Don't do this normally...
##if not os.path.isfile('2017-08-04_00.55.19.pth'):   # 81Mb model
##    wget https://github.com/Cyanogenoid/pytorch-vqa/releases/download/v1.0/2017-08-04_00.55.19.pth
    



#saved_state = torch.load('logs/2017-08-04_00:55:19.pth')
import model # from pytorch-vqa
saved_state = torch.load('2017-08-04_00.55.19.pth', map_location=device)
tokens = len(saved_state['vocab']['question']) + 1
saved_state.keys()  # See what's in the saved state





# Load the predefined model
vqa_net = torch.nn.DataParallel(model.Net(tokens))
vqa_net.load_state_dict(saved_state['weights'])
vqa_net.to(device)
vqa_net.eval()


def get_transform(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Scale(int(target_size / central_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
import resnet  # from pytorch-resnet
class ResNetLayer4(torch.nn.Module):
    def __init__(self):
        super(ResNetLayer4, self).__init__()
        self.model = resnet.resnet152(pretrained=True)
        
        # from  visual_qa_analysis/config.py
        image_size = 448  # scale shorter end of image to this size and centre crop
        #output_size = image_size // 32  # size of the feature maps after processing through a network
        output_features = 2048  # number of feature maps thereof
        central_fraction = 0.875 # only take this much of the centre when scaling and centre cropping

        self.transform = get_transform(image_size, central_fraction)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer
    
    def image_to_features(self, img_file):
        img = Image.open(img_file).convert('RGB')
        img_transformed = self.transform(img)
        #print(img_transformed.size())
        img_batch = img_transformed.unsqueeze(0).to(device)
        return self.forward(img_batch) 
    
resnet_layer4 = ResNetLayer4().to(device)
vocab = saved_state['vocab']
vocab.keys()  # dict_keys(['question', 'answer'])
list(vocab['question'].items())[:5]  # [('the', 1), ('is', 2), ('what', 3), ('are', 4), ('this', 5)]

qtoken_to_index = vocab['question']
QUESTION_LENGTH_MAX = 30 # say...
    
def encode_question(question_str):
    """ Turn a question into a vector of indices and a question length """
    question_arr = question_str.lower().split(' ')
    #vec = torch.zeros(QUESTION_LENGTH_MAX).long()
    vec = torch.zeros(len(question_arr)).long()  
    for i, token in enumerate(question_arr):
        vec[i] = qtoken_to_index.get(token, 0)
    return vec.to(device), torch.tensor( len(question_arr) ).to(device)



list(vocab['answer'].items())[:5]    # [('yes', 0), ('no', 1), ('2', 2), ('1', 3), ('white', 4)]

answer_words = ['UNDEF'] * len(vocab['answer'])
for w,idx in vocab['answer'].items():
    answer_words[idx]=w
len(answer_words), answer_words[:10]  # 3000, ['yes', 'no', '2', '1', 'white', '3', 'red', 'blue', '4', 'green']

# Important things to know...
'colour' in qtoken_to_index, 'color' in qtoken_to_index, 'tabby' in answer_words

image_idx = 1
image_filename = image_files[image_idx]

img = Image.open(image_filename).convert('RGB')
plt.imshow(img)

v0 = resnet_layer4.image_to_features(image_filename)
q, q_len = encode_question("is there a cat in the picture")
#q, q_len = encode_question("what color is the cat's fur")
#q, q_len = encode_question("is the cat jumping up or down")
q, q_len
ans = vqa_net(v0, q.unsqueeze(0), q_len.unsqueeze(0))
ans.data.cpu()[0:10]

_, answer_idx = ans.data.cpu().max(dim=1)
answer_words[ answer_idx ]

def vqa_single_softmax(im_features, q_str):
    q, q_len = encode_question(q_str)
    ans = vqa_net(im_features, q.unsqueeze(0), q_len.unsqueeze(0))
    return ans.data.cpu()

def vqa(image_filename, question_arr):
    plt.imshow(Image.open(image_filename).convert('RGB')); plt.show()    
    image_features = resnet_layer4.image_to_features(image_filename)
    for question_str in question_arr:
        _, answer_idx = vqa_single_softmax(image_features, question_str).max(dim=1)
        #print(question_str+" -> "+answer_words[ answer_idx ])
        print((answer_words[ answer_idx ]+' '*8)[:8]+" <- "+question_str)

print('DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')

def leave_one_out(image_filename, question_base):
    plt.imshow(Image.open(image_filename).convert('RGB')); plt.show()    
    image_features = resnet_layer4.image_to_features(image_filename)
    question_arr = question_base.lower().split(' ')
    for i, word_omit in enumerate(question_arr):
        question_str = ' '.join( question_arr[:i]+question_arr[i+1:] )
        score, answer_idx = vqa_single_softmax(image_features, question_str).max(dim=1)
        #print(question_str+" -> "+answer_words[ answer_idx ])
        print((answer_words[ answer_idx ]+' '*8)[:8]+" <- "+question_str)  #, score




def leave_out_combos(image_filename, question_base):
    plt.imshow(Image.open(image_filename).convert('RGB')); plt.show()    
    image_features = resnet_layer4.image_to_features(image_filename)
    question_arr = question_base.lower().split(' ')
    for i in range(2 ** len(question_arr)):
        q_arr = [question_arr[j] for j in range(len(question_arr)) if (i & (2**j))==0 ]
        question_str = ' '.join( q_arr )
        _, answer_idx = vqa_single_softmax(image_features, question_str).max(dim=1)
        print((answer_words[ answer_idx ]+' '*8)[:8]+" <- "+question_str)


//leave_out_combos("C:\Users\varun\Desktop\demo\VQA\VQA\static\images\8.jpeg", "is women wearing a specticles in the picture")


