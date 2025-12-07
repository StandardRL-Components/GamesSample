
# Generated: 2025-08-27T15:41:04.313115
# Source Brief: brief_01042.md
# Brief Index: 1042

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your character. Catch the green fruits, avoid the red bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game. Catch falling fruits to score points and dodge bombs. The game speeds up over time. Get 50 points to win, but 3 bomb hits and you're out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 16
    CELL_WIDTH = SCREEN_WIDTH // GRID_WIDTH
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_HEIGHT

    # Colors
    COLOR_BG = (25, 25, 40)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (60, 170, 255)
    COLOR_PLAYER_OUTLINE = (200, 230, 255)
    COLOR_FRUIT = (50, 220, 50)
    COLOR_FRUIT_SPECIAL = (255, 220, 0)
    COLOR_BOMB = (255, 60, 60)
    COLOR_BOMB_FUSE = (50, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_OVERLAY = (0, 0, 0, 180)

    # Game Parameters
    INITIAL_FALL_SPEED = 1.0 / 30.0  # grid units per frame
    SPEED_INCREASE_INTERVAL = 300 # frames (10 seconds at 30fps)
    SPEED_INCREASE_AMOUNT = (0.05 / 30.0)
    SPAWN_INTERVAL_MIN = 20
    SPAWN_INTERVAL_MAX = 40
    MAX_STEPS = 1800 # 60 seconds at 30fps
    WIN_SCORE = 50
    MAX_LIVES = 3
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Verdana", 60, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.fruits = None
        self.bombs = None
        self.particles = None
        self.score = None
        self.lives = None
        self.steps = None
        self.game_over = None
        self.fall_speed = None
        self.spawn_timer = None
        self.difficulty_timer = None
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 2]
        self.fruits = []
        self.bombs = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.difficulty_timer = 0
        self.spawn_timer = self.np_random.integers(self.SPAWN_INTERVAL_MIN, self.SPAWN_INTERVAL_MAX)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)
            
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Update Game Logic ---
        reward = self._update_game_state(movement)
        
        self.steps += 1
        
        # --- Check Termination ---
        terminated = (
            self.lives <= 0 
            or self.score >= self.WIN_SCORE 
            or self.steps >= self.MAX_STEPS
        )
        if terminated:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Goal-oriented win reward
            if self.lives <= 0:
                reward -= 100 # Goal-oriented loss reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_game_state(self, movement):
        reward = 0.0

        # 1. Handle Player Movement
        if movement == 0:
            reward -= 0.1 # Penalty for no-op
        elif movement == 1: # Up
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif movement == 2: # Down
            self.player_pos[1] = min(self.GRID_HEIGHT - 1, self.player_pos[1] + 1)
        elif movement == 3: # Left
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif movement == 4: # Right
            self.player_pos[0] = min(self.GRID_WIDTH - 1, self.player_pos[0] + 1)
        
        # 2. Update Falling Objects and Handle Collisions
        # Fruits
        for fruit in self.fruits[:]:
            fruit['pos'][1] += self.fall_speed
            if int(fruit['pos'][0]) == self.player_pos[0] and int(fruit['pos'][1]) == self.player_pos[1]:
                # Sound: fruit_catch.wav
                if fruit['type'] == 'special':
                    self.score += 5
                    reward += 5
                    self._create_particles(fruit['pos'], self.COLOR_FRUIT_SPECIAL, 20)
                else:
                    self.score += 1
                    reward += 1
                    self._create_particles(fruit['pos'], self.COLOR_FRUIT, 10)
                self.fruits.remove(fruit)
            elif fruit['pos'][1] >= self.GRID_HEIGHT:
                self.fruits.remove(fruit)

        # Bombs
        for bomb in self.bombs[:]:
            bomb['pos'][1] += self.fall_speed
            if int(bomb['pos'][0]) == self.player_pos[0] and int(bomb['pos'][1]) == self.player_pos[1]:
                # Sound: explosion.wav
                self.lives -= 1
                self._create_particles(bomb['pos'], self.COLOR_BOMB, 30, is_explosion=True)
                self.bombs.remove(bomb)
            elif bomb['pos'][1] >= self.GRID_HEIGHT:
                self.bombs.remove(bomb)

        # 3. Risk/Reward for being under a bomb
        for bomb in self.bombs:
            if self.player_pos[0] == int(bomb['pos'][0]) and self.player_pos[1] < bomb['pos'][1]:
                reward -= 1.0
                break # Only penalize once per frame

        # 4. Update Particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # 5. Spawn New Objects
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_object()
            self.spawn_timer = self.np_random.integers(self.SPAWN_INTERVAL_MIN, self.SPAWN_INTERVAL_MAX)

        # 6. Increase Difficulty
        self.difficulty_timer += 1
        if self.difficulty_timer >= self.SPEED_INCREASE_INTERVAL:
            self.fall_speed += self.SPEED_INCREASE_AMOUNT
            self.difficulty_timer = 0
            
        return reward

    def _spawn_object(self):
        col = self.np_random.integers(0, self.GRID_WIDTH)
        # 75% chance for fruit, 25% for bomb
        if self.np_random.random() < 0.75:
            # 10% chance for special fruit
            if self.np_random.random() < 0.1:
                self.fruits.append({'pos': [col, 0.0], 'type': 'special'})
            else:
                self.fruits.append({'pos': [col, 0.0], 'type': 'normal'})
        else:
            self.bombs.append({'pos': [col, 0.0]})
            
    def _create_particles(self, pos, color, count, is_explosion=False):
        grid_pos_x = pos[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2
        grid_pos_y = pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        for _ in range(count):
            if is_explosion:
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 5)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.np_random.integers(15, 30)
            else: # Fruit catch effect
                vel = [self.np_random.uniform(-1, 1), self.np_random.uniform(-2, 0)]
                life = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': [grid_pos_x, grid_pos_y],
                'vel': vel,
                'life': life,
                'color': color
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()

        # Render Game Over screen if necessary
        if self.game_over:
            self._render_game_over()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_WIDTH
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py))

        # Draw fruits
        for fruit in self.fruits:
            px = fruit['pos'][0] * self.CELL_WIDTH + self.CELL_WIDTH // 2
            py = fruit['pos'][1] * self.CELL_HEIGHT + self.CELL_HEIGHT // 2
            color = self.COLOR_FRUIT_SPECIAL if fruit['type'] == 'special' else self.COLOR_FRUIT
            radius = self.CELL_WIDTH // 3
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, color)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, color)
            
        # Draw bombs
        for bomb in self.bombs:
            px = bomb['pos'][0] * self.CELL_WIDTH + self.CELL_WIDTH // 2
            py = bomb['pos'][1] * self.CELL_HEIGHT + self.CELL_HEIGHT // 2
            radius = self.CELL_WIDTH // 3
            pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, self.COLOR_BOMB)
            pygame.draw.line(self.screen, self.COLOR_BOMB_FUSE, (int(px), int(py - radius)), (int(px), int(py-radius-4)), 2)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(p['life']/5))

        # Draw player
        player_rect = pygame.Rect(
            self.player_pos[0] * self.CELL_WIDTH,
            self.player_pos[1] * self.CELL_HEIGHT,
            self.CELL_WIDTH,
            self.CELL_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect.inflate(-4, -4))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_OUTLINE, player_rect.inflate(-4, -4), 2)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        bomb_icon_radius = 10
        for i in range(self.lives):
            px = self.SCREEN_WIDTH - 25 - (i * 30)
            py = 22
            pygame.gfxdraw.aacircle(self.screen, px, py, bomb_icon_radius, self.COLOR_BOMB)
            pygame.gfxdraw.filled_circle(self.screen, px, py, bomb_icon_radius, self.COLOR_BOMB)
            pygame.draw.line(self.screen, self.COLOR_BOMB_FUSE, (px, py-bomb_icon_radius), (px, py-bomb_icon_radius-4), 2)
            
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))
        
        if self.score >= self.WIN_SCORE:
            text = "YOU WIN!"
        else:
            text = "GAME OVER"
            
        game_over_surf = self.font_game_over.render(text, True, self.COLOR_TEXT)
        text_rect = game_over_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(game_over_surf, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def close(self):
        pygame.font.quit()
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
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Use a dictionary to track held keys for continuous movement
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Pygame window for human play
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()

    while running:
        movement_action = 0 # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        for key, action_val in key_map.items():
            if keys[key]:
                movement_action = action_val
                break # Prioritize first key found (e.g., up over down)

        # Construct the MultiDiscrete action
        action = [movement_action, 0, 0] # space and shift are not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    env.close()