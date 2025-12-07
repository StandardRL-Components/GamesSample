import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:29:53.550316
# Source Brief: brief_02207.md
# Brief Index: 2207
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Conveyor Belt Sync'.

    The player controls the speed of four conveyor belts to synchronize the arrival
    of different colored items at a central collection point. The goal is to
    collect as many unique item types as possible within a time limit.

    **Action Space (MultiDiscrete([5, 2, 2])):**
    - `action[0]` (Movement): Selects a belt (1:Up, 2:Down, 3:Left, 4:Right). 0 is no-op.
    - `action[1]` (Space): If held (1) and a belt is selected, increases its speed.
    - `action[2]` (Shift): If held (1) and a belt is selected, decreases its speed.
    Note: If both space and shift are held, no speed change occurs.

    **Observation Space:**
    A 640x400x3 RGB image of the game state.

    **Reward:**
    +0.1 reward for each unique item type in a successful collection event (when 2 or more items arrive at the center simultaneously).

    **Termination:**
    The episode ends after 1200 steps (120 seconds of game time).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control the speed of four conveyor belts to synchronize the arrival of items at the center. "
        "Score points by collecting multiple unique items at the same time."
    )
    user_guide = (
        "Controls: Use ↑↓←→ arrow keys to select a belt. "
        "Hold space to increase its speed or shift to decrease its speed."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GAME_LOGIC_HZ = 10
        self.TIME_PER_STEP = 1.0 / self.GAME_LOGIC_HZ
        self.MAX_STEPS = 1200  # 120 seconds * 10 Hz
        self.NUM_BELTS = 4
        self.BELT_SPEED_MIN = 1.0
        self.BELT_SPEED_MAX = 10.0
        self.BELT_WIDTH = 50
        self.ITEM_SIZE = 24
        self.NUM_ITEM_TYPES = 8
        self.ITEMS_PER_BELT = 10
        self.CENTER_X, self.CENTER_Y = self.WIDTH // 2, self.HEIGHT // 2
        self.COLLECTION_RADIUS = self.BELT_WIDTH / 2 + 5

        # Define belt paths and lengths
        self.BELT_LENGTHS = [
            self.CENTER_Y - self.COLLECTION_RADIUS,  # Top
            self.HEIGHT - self.CENTER_Y - self.COLLECTION_RADIUS, # Bottom
            self.CENTER_X - self.COLLECTION_RADIUS,  # Left
            self.WIDTH - self.CENTER_X - self.COLLECTION_RADIUS   # Right
        ]

        # Visuals
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_BELT = (50, 55, 60)
        self.COLOR_BELT_LINE = (80, 85, 90)
        self.COLOR_CENTER = (40, 45, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_FLASH = (255, 255, 255)
        self.ITEM_COLORS = [
            (255, 87, 34),   # Deep Orange
            (3, 169, 244),   # Light Blue
            (255, 235, 59),  # Yellow
            (156, 39, 176),  # Purple
            (76, 175, 80),   # Green
            (233, 30, 99),   # Pink
            (0, 188, 212),   # Cyan
            (255, 193, 7)    # Amber
        ]
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 20)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        # State variables (initialized in reset)
        self.steps = None
        self.belt_speeds = None
        self.items = None
        self.collected_item_types = None
        self.collection_flash_timer = None
        
        # Initialize state
        # self.reset() is called by the wrapper/runner
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.belt_speeds = np.array([self.BELT_SPEED_MIN] * self.NUM_BELTS, dtype=float)
        self.collected_item_types = set()
        self.collection_flash_timer = 0

        self.items = [[] for _ in range(self.NUM_BELTS)]
        for i in range(self.NUM_BELTS):
            # Distribute items somewhat evenly along the belt
            positions = np.linspace(0, self.BELT_LENGTHS[i] * 0.9, self.ITEMS_PER_BELT)
            self.np_random.shuffle(positions)
            for j in range(self.ITEMS_PER_BELT):
                item_type = self.np_random.integers(0, self.NUM_ITEM_TYPES)
                self.items[i].append({
                    "pos": positions[j],
                    "type": item_type,
                    "color": self.ITEM_COLORS[item_type]
                })
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        # --- 1. Update Game Logic based on action ---
        if movement in [1, 2, 3, 4]:
            belt_idx = movement - 1
            # XOR logic: only one button modifies speed at a time
            if space_held and not shift_held:
                # sfx: speed_up_click()
                self.belt_speeds[belt_idx] = min(self.BELT_SPEED_MAX, self.belt_speeds[belt_idx] + 1.0)
            elif shift_held and not space_held:
                # sfx: speed_down_click()
                self.belt_speeds[belt_idx] = max(self.BELT_SPEED_MIN, self.belt_speeds[belt_idx] - 1.0)

        # --- 2. Update Game State ---
        self.steps += 1
        if self.collection_flash_timer > 0:
            self.collection_flash_timer -= 1

        arrived_items = []
        for i in range(self.NUM_BELTS):
            # Speed unit is pixels/sec. We'll scale it for game balance.
            speed_in_pixels = self.belt_speeds[i] * 15
            distance_moved = speed_in_pixels * self.TIME_PER_STEP

            remaining_items = []
            for item in self.items[i]:
                item['pos'] += distance_moved
                if item['pos'] >= self.BELT_LENGTHS[i]:
                    arrived_items.append(item)
                    # sfx: item_arrived_at_center_blip()
                else:
                    remaining_items.append(item)
            self.items[i] = remaining_items

        # --- 3. Calculate Reward ---
        reward = 0.0
        if len(arrived_items) >= 2:
            self.collection_flash_timer = int(self.GAME_LOGIC_HZ / 2) # Flash for 0.5s
            # sfx: collection_success_chime()
            
            unique_types_in_collection = set(item['type'] for item in arrived_items)
            reward = len(unique_types_in_collection) * 0.1
            
            self.collected_item_types.update(unique_types_in_collection)

        # --- 4. Check Termination ---
        terminated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw center collection area
        pygame.draw.circle(self.screen, self.COLOR_CENTER, (self.CENTER_X, self.CENTER_Y), self.COLLECTION_RADIUS)

        # Draw collection flash effect
        if self.collection_flash_timer > 0:
            alpha = int(255 * (self.collection_flash_timer / (self.GAME_LOGIC_HZ / 2)))
            flash_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            radius = self.COLLECTION_RADIUS + 20 * (1 - self.collection_flash_timer / (self.GAME_LOGIC_HZ / 2))
            pygame.gfxdraw.filled_circle(flash_surface, self.CENTER_X, self.CENTER_Y, int(radius), (*self.COLOR_FLASH, alpha))
            self.screen.blit(flash_surface, (0, 0))

        # Belt geometries
        belt_rects = [
            pygame.Rect(self.CENTER_X - self.BELT_WIDTH / 2, 0, self.BELT_WIDTH, self.CENTER_Y - self.COLLECTION_RADIUS),
            pygame.Rect(self.CENTER_X - self.BELT_WIDTH / 2, self.CENTER_Y + self.COLLECTION_RADIUS, self.BELT_WIDTH, self.HEIGHT - (self.CENTER_Y + self.COLLECTION_RADIUS)),
            pygame.Rect(0, self.CENTER_Y - self.BELT_WIDTH / 2, self.CENTER_X - self.COLLECTION_RADIUS, self.BELT_WIDTH),
            pygame.Rect(self.CENTER_X + self.COLLECTION_RADIUS, self.CENTER_Y - self.BELT_WIDTH / 2, self.WIDTH - (self.CENTER_X + self.COLLECTION_RADIUS), self.BELT_WIDTH)
        ]

        # Draw belts and items
        for i in range(self.NUM_BELTS):
            # Draw belt background
            pygame.draw.rect(self.screen, self.COLOR_BELT, belt_rects[i])

            # Draw animated lines on belt for motion
            line_spacing = 40
            offset = (self.steps * self.belt_speeds[i] * 1.5) % line_spacing
            if i < 2: # Vertical belts
                for y_line in range(int(offset), int(belt_rects[i].height), line_spacing):
                    y_pos = belt_rects[i].top + y_line
                    if y_pos < belt_rects[i].bottom:
                        pygame.draw.line(self.screen, self.COLOR_BELT_LINE, (belt_rects[i].left, y_pos), (belt_rects[i].right, y_pos), 2)
            else: # Horizontal belts
                for x_line in range(int(offset), int(belt_rects[i].width), line_spacing):
                    x_pos = belt_rects[i].left + x_line
                    if x_pos < belt_rects[i].right:
                        pygame.draw.line(self.screen, self.COLOR_BELT_LINE, (x_pos, belt_rects[i].top), (x_pos, belt_rects[i].bottom), 2)

            # Draw items on belt
            for item in self.items[i]:
                pos = item['pos']
                if i == 0: x, y = self.CENTER_X, pos                                  # Top
                elif i == 1: x, y = self.CENTER_X, self.HEIGHT - pos                  # Bottom
                elif i == 2: x, y = pos, self.CENTER_Y                                # Left
                else: x, y = self.WIDTH - pos, self.CENTER_Y                         # Right
                
                item_rect = pygame.Rect(x - self.ITEM_SIZE / 2, y - self.ITEM_SIZE / 2, self.ITEM_SIZE, self.ITEM_SIZE)
                pygame.draw.rect(self.screen, item['color'], item_rect, border_radius=4)
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in item['color']), item_rect, width=2, border_radius=4)

    def _render_ui(self):
        # --- Score and Time ---
        score_text = self.font_large.render(f"UNIQUE ITEMS: {len(self.collected_item_types)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH / 2 - score_text.get_width() / 2, 10))
        
        time_left = (self.MAX_STEPS - self.steps) * self.TIME_PER_STEP
        time_text = self.font_medium.render(f"TIME: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 20, 15))

        # --- Belt Speed Indicators ---
        speed_ui_positions = [
            (self.CENTER_X, 30, 'v'), # Top
            (self.CENTER_X, self.HEIGHT - 50, 'v'), # Bottom
            (30, self.CENTER_Y, 'h'), # Left
            (self.WIDTH - 130, self.CENTER_Y, 'h') # Right
        ]
        belt_labels = ["▲ UP", "▼ DOWN", "◀ LEFT", "▶ RIGHT"]

        for i in range(self.NUM_BELTS):
            x, y, orientation = speed_ui_positions[i]
            speed = self.belt_speeds[i]
            
            label_text = self.font_small.render(belt_labels[i], True, self.COLOR_UI_TEXT)
            speed_text = self.font_medium.render(f"SPD: {int(speed)}", True, self.COLOR_UI_TEXT)
            
            bar_width = 100
            bar_height = 15
            fill_ratio = (speed - self.BELT_SPEED_MIN) / (self.BELT_SPEED_MAX - self.BELT_SPEED_MIN)

            if orientation == 'v':
                self.screen.blit(label_text, (x - label_text.get_width()/2, y))
                self.screen.blit(speed_text, (x - speed_text.get_width()/2, y + 20))
                
                bar_rect = pygame.Rect(x - bar_width/2, y + 45, bar_width, bar_height)
                fill_rect = pygame.Rect(x - bar_width/2, y + 45, bar_width * fill_ratio, bar_height)
            else: # 'h'
                self.screen.blit(label_text, (x, y - 30))
                self.screen.blit(speed_text, (x, y - 10))

                bar_rect = pygame.Rect(x, y + 15, bar_width, bar_height)
                fill_rect = pygame.Rect(x, y + 15, bar_width * fill_ratio, bar_height)

            pygame.draw.rect(self.screen, self.COLOR_BELT, bar_rect, border_radius=3)
            if fill_rect.width > 0:
                pygame.draw.rect(self.screen, self.ITEM_COLORS[1], fill_rect, border_radius=3)

    def _get_info(self):
        return {
            "score": len(self.collected_item_types),
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        # This is a helper for development and can be removed.
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # We need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Conveyor Belt Sync")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        if terminated:
            print(f"Episode finished. Final Score: {info['score']}")
            obs, info = env.reset()
            terminated = False

        # --- Human Controls ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            print(f"Step {info['steps']}: Reward of {reward:.2f}, New Score: {info['score']}")

        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        # --- Rendering ---
        # The observation is already a rendered frame, so we just display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the human play speed to be similar to the agent's
        clock.tick(env.GAME_LOGIC_HZ)

    env.close()