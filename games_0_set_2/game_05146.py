
# Generated: 2025-08-28T04:06:07.323299
# Source Brief: brief_05146.md
# Brief Index: 5146

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Collect fruit and avoid the bombs."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade game. Collect fruit to score points while dodging bombs. Risky collections near bombs yield bonus points. Collect 50 fruits to win!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.WIN_SCORE = 50
        self.MAX_STEPS = 1500 # Increased to allow more time to reach win score
        
        # Player settings
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 5
        
        # Item settings
        self.FRUIT_SIZE = 8
        self.BOMB_SIZE = 10
        self.FRUIT_SPAWN_RATE = 25
        self.BOMB_SPAWN_RATE = 50
        self.FRUIT_LIFESPAN = 150 # Increased lifespan for better playability
        self.BOMB_LIFESPAN = 200
        self.RISKY_DISTANCE = 50
        
        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_BOMB = (255, 50, 50)
        self.COLOR_BOMB_SKULL = (220, 220, 220)
        self.COLOR_BOMB_EYES = (10, 10, 10)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_BONUS_TEXT = (255, 220, 0)
        self.FRUIT_COLORS = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (100, 255, 255), (255, 100, 255)
        ]

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_bonus = pygame.font.Font(None, 24)
        
        # State variables (initialized in reset)
        self.np_random = None
        self.player_pos = None
        self.fruits = None
        self.bombs = None
        self.bonus_texts = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        # Initialize state
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize RNG
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        # Initialize game state
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.fruits = []
        self.bombs = []
        self.bonus_texts = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        # Initial spawn: 5 safe fruits, 3 random bombs
        for _ in range(5):
             self._spawn_item(self.fruits, self.FRUIT_SIZE, safe_spawn=True)
        for _ in range(3):
            self._spawn_item(self.bombs, self.BOMB_SIZE)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        
        self.steps += 1
        reward = -0.01 # Small time penalty to encourage action

        # 1. Update Player Position
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # 2. Update Items (Age, Despawn)
        self.fruits = [f for f in self.fruits if f['age'] < self.FRUIT_LIFESPAN]
        for f in self.fruits: f['age'] += 1
        
        self.bombs = [b for b in self.bombs if b['age'] < self.BOMB_LIFESPAN]
        for b in self.bombs: b['age'] += 1

        self.bonus_texts = [t for t in self.bonus_texts if t['age'] < 60]
        for t in self.bonus_texts:
            t['pos'][1] -= 0.5
            t['age'] += 1

        # 3. Handle Collisions
        # Fruit collection
        collected_indices = []
        for i, fruit in enumerate(self.fruits):
            dist = np.linalg.norm(self.player_pos - fruit['pos'])
            if dist < self.PLAYER_SIZE + self.FRUIT_SIZE:
                # sfx: fruit_collect.wav
                collected_indices.append(i)
                self.score += 1
                reward += 1.0
                
                # Check for risky collection bonus
                min_dist_to_bomb = float('inf')
                if self.bombs:
                    for bomb in self.bombs:
                        dist_to_bomb = np.linalg.norm(fruit['pos'] - bomb['pos'])
                        min_dist_to_bomb = min(min_dist_to_bomb, dist_to_bomb)
                
                if min_dist_to_bomb < self.RISKY_DISTANCE:
                    # sfx: bonus_point.wav
                    reward += 5.0
                    self.score += 5 # Bonus points also add to score
                    self.bonus_texts.append({'pos': list(fruit['pos']), 'text': '+6', 'age': 0})
                else:
                    self.bonus_texts.append({'pos': list(fruit['pos']), 'text': '+1', 'age': 0})

        if collected_indices:
            self.fruits = [f for i, f in enumerate(self.fruits) if i not in collected_indices]

        # Bomb collision
        for bomb in self.bombs:
            dist = np.linalg.norm(self.player_pos - bomb['pos'])
            if dist < self.PLAYER_SIZE + self.BOMB_SIZE:
                # sfx: explosion.wav
                self.game_over = True
                reward = -100.0
                break

        # 4. Spawn New Items
        if self.steps % self.FRUIT_SPAWN_RATE == 0:
            self._spawn_item(self.fruits, self.FRUIT_SIZE)
        if self.steps % self.BOMB_SPAWN_RATE == 0:
            self._spawn_item(self.bombs, self.BOMB_SIZE)
            
        # 5. Check Termination Conditions
        terminated = self.game_over or self.score >= self.WIN_SCORE or self.steps >= self.MAX_STEPS
        if not self.game_over and self.score >= self.WIN_SCORE:
            # sfx: game_win.wav
            reward = 100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_item(self, item_list, size, safe_spawn=False):
        for _ in range(10): # Try 10 times to find a free spot
            if safe_spawn: # Spawn in a safer central area
                pos = self.np_random.integers(
                    [self.WIDTH * 0.2, self.HEIGHT * 0.2],
                    [self.WIDTH * 0.8, self.HEIGHT * 0.8],
                    size=2).astype(np.float32)
            else:
                pos = self.np_random.integers(
                    [size, size], 
                    [self.WIDTH - size, self.HEIGHT - size], 
                    size=2).astype(np.float32)

            # Check collision with player
            if np.linalg.norm(pos - self.player_pos) < self.PLAYER_SIZE * 3:
                continue
            
            # Check collision with other items
            is_overlapping = False
            all_items = self.fruits + self.bombs
            for item in all_items:
                if np.linalg.norm(pos - item['pos']) < item['size'] * 3:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                item_data = {'pos': pos, 'size': size, 'age': 0}
                if item_list is self.fruits:
                    item_data['color'] = random.choice(self.FRUIT_COLORS)
                item_list.append(item_data)
                return

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render bombs
        for bomb in self.bombs:
            pos = bomb['pos'].astype(int)
            size = int(bomb['size'])
            
            # Blinking effect
            blink_alpha = 128 + 127 * math.sin(bomb['age'] * 0.1 + bomb['pos'][0])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, self.COLOR_BOMB)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, self.COLOR_BOMB)
            
            # Skull icon
            skull_size = int(size * 0.7)
            eye_size = int(size * 0.15)
            pygame.draw.circle(self.screen, self.COLOR_BOMB_SKULL, pos, skull_size)
            pygame.draw.circle(self.screen, self.COLOR_BOMB_EYES, (pos[0] - skull_size//3, pos[1] - skull_size//4), eye_size)
            pygame.draw.circle(self.screen, self.COLOR_BOMB_EYES, (pos[0] + skull_size//3, pos[1] - skull_size//4), eye_size)


        # Render fruits
        for fruit in self.fruits:
            pos = fruit['pos'].astype(int)
            size = int(fruit['size'])
            
            # Pulsing effect
            pulse = 1 + 0.2 * math.sin(fruit['age'] * 0.2 + fruit['pos'][1])
            current_size = int(size * pulse)
            
            rect = pygame.Rect(pos[0] - current_size, pos[1] - current_size, current_size * 2, current_size * 2)
            pygame.draw.rect(self.screen, fruit['color'], rect, border_radius=3)

        # Render player
        player_pos_int = self.player_pos.astype(int)
        glow_size = int(self.PLAYER_SIZE * 2.5)
        glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (glow_size, glow_size), glow_size)
        self.screen.blit(glow_surf, (player_pos_int[0] - glow_size, player_pos_int[1] - glow_size), special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)

        # Render bonus texts
        for t in self.bonus_texts:
            alpha = max(0, 255 - (t['age'] * 4))
            text_surf = self.font_bonus.render(t['text'], True, self.COLOR_BONUS_TEXT)
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, (int(t['pos'][0]), int(t['pos'][1])))

    def _render_ui(self):
        score_text = f"Score: {self.score} / {self.WIN_SCORE}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        steps_text = f"Steps: {self.steps} / {self.MAX_STEPS}"
        steps_surf = self.font_ui.render(steps_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_surf, (self.WIDTH - steps_surf.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_pos": self.player_pos,
            "fruit_count": len(self.fruits),
            "bomb_count": len(self.bombs),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    # Test the environment
    env = GameEnv()
    obs, info = env.reset()
    print("Initial state:", info)
    
    terminated = False
    total_reward = 0
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            print("Episode terminated.")
            break
    
    print("Final state after 200 random steps:", info)
    print("Total reward:", total_reward)

    # Example of how to render and play manually
    # Note: This requires a display. Comment out the os.environ line above.
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Fruit Dodger")
    # clock = pygame.time.Clock()
    # running = True
    # while running:
    #     action = [0, 0, 0] # Default no-op
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False

    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: action[0] = 1
    #     elif keys[pygame.K_DOWN]: action[0] = 2
    #     elif keys[pygame.K_LEFT]: action[0] = 3
    #     elif keys[pygame.K_RIGHT]: action[0] = 4

    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     # Convert obs back to a Pygame surface to display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()

    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         running = False
    #         pygame.time.wait(2000)

    #     clock.tick(30) # Match auto_advance rate
    
    # env.close()