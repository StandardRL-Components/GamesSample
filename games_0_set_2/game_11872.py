import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:39:43.498885
# Source Brief: brief_01872.md
# Brief Index: 1872
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a rhythmic block stacking game.

    **Objective:** Stack 10 blocks to build a stable tower.
    **Scoring:** Points are awarded for placing blocks, achieving new heights,
                 and synchronizing drops with the game's rhythm.
    **Failure:** The tower collapses if a block is placed too unstably.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Stack blocks in time with the beat to build a stable tower. Place blocks accurately and on-rhythm "
        "to score points and reach the target height."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move the block. Hold space to drop it faster."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAY_AREA_WIDTH = 240
    FLOOR_HEIGHT = 20
    BLOCK_SIZE = 30
    TARGET_HEIGHT = 10
    MAX_STEPS = 5000
    PLAYER_SPEED = 6.0
    BASE_FALL_SPEED_INITIAL = 1.0
    FAST_FALL_SPEED = 8.0
    BPM = 120
    FPS = 30 # Assumed FPS for rhythm calculation

    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_GRID = (30, 35, 58)
    COLOR_FLOOR = (50, 60, 85)
    COLOR_UI_TEXT = (230, 230, 255)
    COLOR_UI_PANEL = (25, 30, 50, 200)
    COLOR_TARGET_LINE = (255, 200, 0)
    COLOR_SHADOW = (0, 0, 0, 100)
    SHAPE_COLORS = [
        (255, 87, 87),   # Red
        (87, 189, 255),  # Blue
        (87, 255, 154),  # Green
        (255, 170, 87),  # Orange
        (204, 87, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_title = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_reward = pygame.font.SysFont("Consolas", 16, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.stacked_blocks = None
        self.falling_block = None
        self.particles = None
        self.base_fall_speed = None
        self.fall_speed = None
        self.rhythm_bonus_multiplier = None
        self.rhythm_flash_timer = None
        self.height_record = None
        self.last_reward_info = None

        self.BPM_FRAMES = (60 / self.BPM) * self.FPS
        self.play_area_left = (self.SCREEN_WIDTH - self.PLAY_AREA_WIDTH) / 2
        self.play_area_right = self.play_area_left + self.PLAY_AREA_WIDTH
        
        # This call is not strictly necessary but good practice
        # self.reset() # reset is called by the wrapper or user
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.stacked_blocks = []
        self.particles = []
        self.base_fall_speed = self.BASE_FALL_SPEED_INITIAL
        self.fall_speed = self.base_fall_speed
        self.rhythm_bonus_multiplier = 1
        self.rhythm_flash_timer = 0
        self.height_record = 0
        self.last_reward_info = {"value": 0, "timer": 0}

        self._create_floor()
        self._spawn_new_block()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False

        self._handle_input(action)
        self._update_game_state()

        # --- Collision and Placement Logic ---
        top_block = self.stacked_blocks[-1]
        if self.falling_block['y'] + self.BLOCK_SIZE >= top_block['y']:
            # --- Block has landed ---
            self.falling_block['y'] = top_block['y'] - self.BLOCK_SIZE
            placement_reward = 0.1
            reward += placement_reward
            # sound: place_block.wav

            # --- Check Stability ---
            # Collapse if horizontal overlap is less than 20%
            top_rect = pygame.Rect(top_block['x'], top_block['y'], top_block.get('width', self.BLOCK_SIZE), top_block.get('height', self.BLOCK_SIZE))
            new_rect = pygame.Rect(self.falling_block['x'], self.falling_block['y'], self.BLOCK_SIZE, self.BLOCK_SIZE)
            
            overlap = max(0, min(top_rect.right, new_rect.right) - max(top_rect.left, new_rect.left))

            if overlap < self.BLOCK_SIZE * 0.2 and len(self.stacked_blocks) > 1:
                # --- Collapse ---
                reward = -100.0
                terminated = True
                self.game_over = True
                self._create_collapse_particles(self.falling_block)
                # sound: collapse.wav
            else:
                # --- Stable Placement ---
                # Rhythm Bonus
                beat_frame = self.steps % self.BPM_FRAMES
                if beat_frame <= 1 or beat_frame >= self.BPM_FRAMES - 1:
                    rhythm_reward = 0.5 * self.rhythm_bonus_multiplier
                    reward += rhythm_reward
                    self.rhythm_bonus_multiplier = min(5, self.rhythm_bonus_multiplier + 1)
                    self.rhythm_flash_timer = 4 # Flash for 4 frames
                    # sound: rhythm_success.wav
                else:
                    self.rhythm_bonus_multiplier = 1 # Reset multiplier on miss

                self.stacked_blocks.append(self.falling_block)
                
                # Height Reward
                current_height = len(self.stacked_blocks) - 1
                if current_height > self.height_record:
                    height_reward = 1.0
                    reward += height_reward
                    self.height_record = current_height

                # Win Condition
                if current_height >= self.TARGET_HEIGHT:
                    win_reward = 100.0
                    reward = win_reward # Override other rewards for the final one
                    terminated = True
                    self.game_over = True
                    # sound: victory.wav
                else:
                    self._spawn_new_block()

        # --- Other Termination Conditions ---
        if self.steps >= self.MAX_STEPS:
            truncated = True
            terminated = True # Per Gymnasium API, truncated episodes are also terminated
            self.game_over = True

        self.score += reward
        if reward != 0:
            self.last_reward_info = {"value": reward, "timer": self.FPS} # Show for 1 sec

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        # shift_held = action[2] == 1 # Unused

        # Horizontal movement (3=left, 4=right)
        if movement == 3:
            self.falling_block['x'] -= self.PLAYER_SPEED
        elif movement == 4:
            self.falling_block['x'] += self.PLAYER_SPEED
        
        self.falling_block['x'] = np.clip(
            self.falling_block['x'], self.play_area_left, self.play_area_right - self.BLOCK_SIZE
        )

        # Fast drop
        self.fall_speed = self.FAST_FALL_SPEED if space_held else self.base_fall_speed

    def _update_game_state(self):
        # Update falling block position
        if not self.game_over:
            self.falling_block['y'] += self.fall_speed

        # Update difficulty (fall speed)
        if self.steps > 0 and self.steps % 500 == 0:
            self.base_fall_speed = min(self.base_fall_speed + 0.05, self.BLOCK_SIZE / 4)

        # Update visual effect timers
        if self.rhythm_flash_timer > 0:
            self.rhythm_flash_timer -= 1
        if self.last_reward_info['timer'] > 0:
            self.last_reward_info['timer'] -= 1

        # Update particles
        self._update_particles()

    def _spawn_new_block(self):
        start_x = self.play_area_left + (self.PLAY_AREA_WIDTH / 2) - (self.BLOCK_SIZE / 2)
        self.falling_block = {
            'x': start_x,
            'y': -self.BLOCK_SIZE,
            'color': random.choice(self.SHAPE_COLORS)
        }

    def _create_floor(self):
        self.stacked_blocks.append({
            'x': 0,
            'y': self.SCREEN_HEIGHT - self.FLOOR_HEIGHT,
            'width': self.SCREEN_WIDTH,
            'height': self.FLOOR_HEIGHT,
            'color': self.COLOR_FLOOR
        })

    def _create_collapse_particles(self, block):
        # sound: particle_burst.wav
        for _ in range(30):
            self.particles.append({
                'pos': pygame.Vector2(block['x'] + self.BLOCK_SIZE / 2, block['y'] + self.BLOCK_SIZE / 2),
                'vel': pygame.Vector2(random.uniform(-4, 4), random.uniform(-6, 1)),
                'life': random.randint(20, 40),
                'radius': random.uniform(2, 6),
                'color': block['color']
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.2  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background grid
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

        # Target height line
        target_y = self.SCREEN_HEIGHT - self.FLOOR_HEIGHT - (self.TARGET_HEIGHT * self.BLOCK_SIZE)
        if target_y > 0:
            for x in range(int(self.play_area_left), int(self.play_area_right), 20):
                 pygame.draw.line(self.screen, self.COLOR_TARGET_LINE, (x, target_y), (x + 10, target_y), 2)

        # Stacked blocks
        for block in self.stacked_blocks:
            rect = pygame.Rect(block['x'], block['y'], block.get('width', self.BLOCK_SIZE), block.get('height', self.BLOCK_SIZE))
            pygame.draw.rect(self.screen, (0,0,0), rect.inflate(4, 4))
            pygame.draw.rect(self.screen, block['color'], rect)

        # Falling block and shadow
        if self.falling_block and not self.game_over:
            # Shadow
            shadow_y = self.stacked_blocks[-1]['y'] - self.BLOCK_SIZE
            shadow_rect = pygame.Rect(self.falling_block['x'], shadow_y, self.BLOCK_SIZE, self.BLOCK_SIZE)
            shadow_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            shadow_surface.fill(self.COLOR_SHADOW)
            self.screen.blit(shadow_surface, shadow_rect.topleft)

            # Block
            block_rect = pygame.Rect(self.falling_block['x'], self.falling_block['y'], self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, (0,0,0), block_rect.inflate(4, 4))
            pygame.draw.rect(self.screen, self.falling_block['color'], block_rect)
        
        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), int(p['radius'] * (p['life'] / 40.0)))
        
        # Rhythm flash effect
        if self.rhythm_flash_timer > 0:
            flash_alpha = 150 * (self.rhythm_flash_timer / 4.0)
            flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, flash_alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # UI Panel
        panel_rect = pygame.Rect(10, 10, 180, 120)
        panel_surface = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel_surface.fill(self.COLOR_UI_PANEL)
        self.screen.blit(panel_surface, panel_rect.topleft)
        
        # Score
        score_text = self.font_title.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Height
        height = max(0, len(self.stacked_blocks) - 1)
        height_text = self.font_ui.render(f"Height: {height} / {self.TARGET_HEIGHT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(height_text, (25, 55))

        # Rhythm Multiplier
        mult_text = self.font_ui.render(f"Rhythm: x{self.rhythm_bonus_multiplier}", True, self.COLOR_UI_TEXT)
        self.screen.blit(mult_text, (25, 80))
        
        # Rhythm beat indicator
        beat_progress = (self.steps % self.BPM_FRAMES) / self.BPM_FRAMES
        beat_bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_GRID, (25, 105, beat_bar_width, 10))
        beat_marker_pos = 25 + beat_progress * beat_bar_width
        pygame.draw.rect(self.screen, self.COLOR_TARGET_LINE, (beat_marker_pos - 2, 102, 4, 16))
        
        # Last reward indicator
        if self.last_reward_info['timer'] > 0:
            val = self.last_reward_info['value']
            reward_str = f"+{val:.1f}" if val > 0 else f"{val:.1f}"
            color = (100, 255, 100) if val > 0 else (255, 100, 100)
            reward_text = self.font_reward.render(reward_str, True, color)
            alpha = 255 * (self.last_reward_info['timer'] / self.FPS)
            reward_text.set_alpha(alpha)
            self.screen.blit(reward_text, (25, 135))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "height": max(0, len(self.stacked_blocks) - 1)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Rhythmic Stacker")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    while not terminated:
        # --- Human Controls ---
        movement = 0 # 0=none, 3=left, 4=right
        space_held = 0
        shift_held = 0 # unused

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is the frame, so we just need to show it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
    env.close()