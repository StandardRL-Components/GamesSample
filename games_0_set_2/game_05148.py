
# Generated: 2025-08-28T04:07:13.046867
# Source Brief: brief_05148.md
# Brief Index: 5148

        
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

    user_guide = (
        "Controls: Use arrow keys to move your monster on the grid. "
        "Your goal is to eat food and reach 100 points before the timer runs out."
    )

    game_description = (
        "A fast-paced, grid-based arcade game. Navigate a monster to devour food and "
        "reach a target score. Each stage presents new challenges with more obstacles. "
        "Blue food is worth 2 points, Yellow is 5, and Red is 10. Choose your path wisely!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 30, 18
        self.CELL_SIZE = 20
        self.X_OFFSET = (self.WIDTH - self.GRID_W * self.CELL_SIZE) // 2
        self.Y_OFFSET = (self.HEIGHT - self.GRID_H * self.CELL_SIZE) // 2
        
        self.WIN_SCORE = 100
        self.MAX_STEPS = 180 # 3 stages * 60 steps/stage
        self.STAGE_DURATION = 60
        self.FOOD_SPAWN_INTERVAL = 20
        self.INITIAL_FOOD_COUNT = 15
        self.BASE_OBSTACLE_DENSITY = 0.10

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_PLAYER = (0, 255, 127) # Spring Green
        self.COLOR_OBSTACLE = (105, 105, 105) # Dim Gray
        self.COLOR_FOOD_BLUE = (0, 191, 255) # Deep Sky Blue
        self.COLOR_FOOD_YELLOW = (255, 215, 0) # Gold
        self.COLOR_FOOD_RED = (255, 69, 0) # OrangeRed
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_small = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_large = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_small = pygame.font.SysFont("sans", 18)
            self.font_large = pygame.font.SysFont("sans", 24)

        # --- Game State Initialization ---
        self.player_pos = None
        self.food = None
        self.obstacles = None
        self.score = 0
        self.steps = 0
        self.timer = 0
        self.stage = 1
        self.np_random = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.timer = self.MAX_STEPS
        
        self._generate_level(self.stage)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        reward = 0
        
        # --- Handle Movement ---
        prev_pos = self.player_pos
        target_pos = list(self.player_pos)
        
        if movement == 1: # Up
            target_pos[1] -= 1
        elif movement == 2: # Down
            target_pos[1] += 1
        elif movement == 3: # Left
            target_pos[0] -= 1
        elif movement == 4: # Right
            target_pos[0] += 1

        target_pos = tuple(target_pos)

        # --- Collision & State Update ---
        if movement == 0: # No-op
            reward -= 0.2
        elif not (0 <= target_pos[0] < self.GRID_W and 0 <= target_pos[1] < self.GRID_H) or target_pos in self.obstacles:
            # Hit wall or obstacle
            reward -= 0.2
        else:
            self.player_pos = target_pos
        
        # --- Food Consumption ---
        consumed_food_index = -1
        for i, f in enumerate(self.food):
            if self.player_pos == f['pos']:
                # Score update based on game rules
                self.score += f['points']
                # Reward update based on RL structure
                reward += f['reward']
                consumed_food_index = i
                # sfx: play_consume_sound()
                break
        
        if consumed_food_index != -1:
            self.food.pop(consumed_food_index)

        # --- Update Timers & Spawners ---
        self.steps += 1
        self.timer -= 1
        
        if self.steps > 0 and self.steps % self.FOOD_SPAWN_INTERVAL == 0:
            self._spawn_food()

        # --- Stage Progression ---
        if self.steps > 0 and self.steps % self.STAGE_DURATION == 0 and self.steps < self.MAX_STEPS:
            if self.score < self.WIN_SCORE:
                self.stage += 1
                self._generate_level(self.stage)

        # --- Termination Check ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            # sfx: play_win_sound()
        elif self.timer <= 0:
            reward -= 100
            terminated = True
            # sfx: play_lose_sound()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self, stage):
        self.obstacles = set()
        self.food = []

        # Get all possible grid coordinates
        all_cells = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(all_cells)

        # Place player
        self.player_pos = all_cells.pop()
        
        # Place obstacles
        obstacle_density = self.BASE_OBSTACLE_DENSITY + (stage - 1) * 0.10
        num_obstacles = int(self.GRID_W * self.GRID_H * obstacle_density)
        for _ in range(min(num_obstacles, len(all_cells))):
            self.obstacles.add(all_cells.pop())
            
        # Place initial food
        food_types = [
            {'color': self.COLOR_FOOD_BLUE, 'points': 2, 'reward': 1},
            {'color': self.COLOR_FOOD_YELLOW, 'points': 5, 'reward': 2},
            {'color': self.COLOR_FOOD_RED, 'points': 10, 'reward': 5}
        ]
        
        for _ in range(min(self.INITIAL_FOOD_COUNT, len(all_cells))):
            pos = all_cells.pop()
            food_type = self.np_random.choice(food_types, p=[0.6, 0.3, 0.1])
            self.food.append({'pos': pos, **food_type})

    def _spawn_food(self):
        occupied_cells = self.obstacles.union({self.player_pos}, {f['pos'] for f in self.food})
        empty_cells = []
        for x in range(self.GRID_W):
            for y in range(self.GRID_H):
                if (x, y) not in occupied_cells:
                    empty_cells.append((x, y))

        if not empty_cells:
            return

        pos = self.np_random.choice(empty_cells)
        pos = (pos[0], pos[1]) # convert from np array if needed

        food_types = [
            {'color': self.COLOR_FOOD_BLUE, 'points': 2, 'reward': 1},
            {'color': self.COLOR_FOOD_YELLOW, 'points': 5, 'reward': 2},
            {'color': self.COLOR_FOOD_RED, 'points': 10, 'reward': 5}
        ]
        food_type = self.np_random.choice(food_types, p=[0.6, 0.3, 0.1])
        self.food.append({'pos': pos, **food_type})
        # sfx: play_spawn_sound()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_W + 1):
            px = self.X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.Y_OFFSET), (px, self.Y_OFFSET + self.GRID_H * self.CELL_SIZE))
        for y in range(self.GRID_H + 1):
            py = self.Y_OFFSET + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.X_OFFSET, py), (self.X_OFFSET + self.GRID_W * self.CELL_SIZE, py))

        # Draw obstacles
        for ox, oy in self.obstacles:
            rect = pygame.Rect(
                self.X_OFFSET + ox * self.CELL_SIZE,
                self.Y_OFFSET + oy * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw food
        for f in self.food:
            pos_x, pos_y = f['pos']
            center_x = self.X_OFFSET + int((pos_x + 0.5) * self.CELL_SIZE)
            center_y = self.Y_OFFSET + int((pos_y + 0.5) * self.CELL_SIZE)
            radius = int(self.CELL_SIZE * 0.35)
            
            # Pulsing effect
            pulse = (math.sin(self.steps * 0.2 + pos_x) + 1) / 2
            current_radius = int(radius * (0.9 + 0.2 * pulse))
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, current_radius, f['color'])
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, current_radius, f['color'])

        # Draw player
        px, py = self.player_pos
        center_x = self.X_OFFSET + int((px + 0.5) * self.CELL_SIZE)
        center_y = self.Y_OFFSET + int((py + 0.5) * self.CELL_SIZE)
        
        # Breathing animation
        pulse = (math.sin(self.steps * 0.3) + 1) / 2
        outer_radius = int(self.CELL_SIZE * 0.4 * (1.0 + 0.2 * pulse))
        inner_radius = int(self.CELL_SIZE * 0.4)
        
        # Glow effect
        glow_color = list(self.COLOR_PLAYER)
        glow_color.append(int(70 + 80 * pulse)) # Alpha
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, outer_radius, glow_color)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, outer_radius, glow_color)
        
        # Main body
        pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, inner_radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, center_x, center_y, inner_radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", (20, 15), self.font_large, align="topleft")
        # Timer
        self._render_text(f"TIME: {self.timer}", (self.WIDTH - 20, 15), self.font_large, align="topright")
        # Stage
        self._render_text(f"STAGE: {self.stage}", (self.WIDTH // 2, self.HEIGHT - 20), self.font_large, align="midbottom")

    def _render_text(self, text, position, font, align="center"):
        # Shadow
        text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect_shadow = text_surf_shadow.get_rect()
        setattr(text_rect_shadow, align, (position[0] + 2, position[1] + 2))
        self.screen.blit(text_surf_shadow, text_rect_shadow)
        
        # Main Text
        text_surf = font.render(text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect()
        setattr(text_rect, align, position)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "stage": self.stage,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a display for human play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid Monster")
    
    action = env.action_space.sample()
    action[0] = 0 # Start with no-op

    print("\n" + "="*30)
    print("      GRID MONSTER      ")
    print("="*30)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Human controls
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, 0, 0] # Space and shift are not used

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Action: {action[0]}, Reward: {reward:.1f}, Score: {info['score']}, Terminated: {terminated}")
        else:
            print(f"--- GAME OVER ---")
            print(f"Final Score: {info['score']} in {info['steps']} steps.")
            if info['score'] >= env.WIN_SCORE:
                print("Result: YOU WIN!")
            else:
                print("Result: TIME'S UP!")
            print("Resetting in 3 seconds...")
            pygame.time.wait(3000)
            obs, info = env.reset()
            terminated = False

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(10) # Control human play speed
        
    env.close()