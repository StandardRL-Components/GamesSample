import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:57:05.044585
# Source Brief: brief_01424.md
# Brief Index: 1424
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Conveyor Belt Sort
    A Gymnasium environment where the agent controls the speed of two conveyor
    belts to sort colored items into their corresponding bins.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control the speed of two conveyor belts to sort colored items into their corresponding bins before they fall off or are sorted incorrectly."
    )
    user_guide = (
        "Controls: Use ↑/↓ to adjust the top belt speed and ←/→ for the bottom belt. "
        "Hold Space to stop the top belt and Shift to stop the bottom belt."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TOTAL_ITEMS = 15
    MAX_STEPS = 1500 # Increased for slower strategies

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_BELT = (50, 60, 70)
    COLOR_BELT_SHADOW = (15, 25, 35)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_SPEED_BAR_BG = (40, 50, 60)
    COLOR_SPEED_BAR_FILL = (66, 175, 241) # A bright cyan
    COLOR_ITEM_RED = (255, 70, 70)
    COLOR_ITEM_GREEN = (70, 255, 70)
    COLOR_ITEM_BLUE = (70, 130, 255)
    ITEM_COLORS = [COLOR_ITEM_RED, COLOR_ITEM_GREEN, COLOR_ITEM_BLUE]

    # Game physics and layout
    BELT_Y_TOP = 120
    BELT_Y_BOTTOM = 280
    BELT_HEIGHT = 60
    BELT_START_X = -100
    BELT_END_X = SCREEN_WIDTH + 50
    BELT_TRANSFER_X = 400
    MAX_PIXELS_PER_STEP = 8.0
    ITEM_RADIUS = 12
    BIN_WIDTH = 80
    BIN_Y = BELT_Y_BOTTOM
    BIN_START_X = SCREEN_WIDTH - (BIN_WIDTH * 3)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.top_belt_speed = 0.0
        self.bottom_belt_speed = 0.0
        self.items = []
        self.particles = []
        self.belt_scroll_top = 0
        self.belt_scroll_bottom = 0

        # Bin setup
        self.bins = []
        for i, color in enumerate(self.ITEM_COLORS):
            self.bins.append({
                'rect': pygame.Rect(self.BIN_START_X + i * self.BIN_WIDTH, self.BIN_Y, self.BIN_WIDTH, self.BELT_HEIGHT),
                'color': color
            })
        
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.top_belt_speed = 0.2
        self.bottom_belt_speed = 0.2
        self.particles = []
        self.belt_scroll_top = 0
        self.belt_scroll_bottom = 0

        # Create items
        self.items = []
        item_colors_shuffled = [self.ITEM_COLORS[self.np_random.integers(0, 3)] for _ in range(self.TOTAL_ITEMS)]
        for i in range(self.TOTAL_ITEMS):
            start_x = self.BELT_START_X - i * (self.ITEM_RADIUS * 3)
            self.items.append({
                'pos': pygame.Vector2(start_x, self.BELT_Y_TOP),
                'color': item_colors_shuffled[i],
                'belt_idx': 0, # 0 for top, 1 for bottom
                'id': i,
                'sorted': False
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_actions(action)
        reward += self._update_game_state()
        self.steps += 1

        # Check termination conditions
        all_sorted = all(item['sorted'] for item in self.items)
        if all_sorted:
            reward += 100.0 # Victory bonus
            terminated = True
        elif self.game_over: # Set by _update_game_state on error
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        speed_change = 0.05

        # Top belt control
        if space_held:
            self.top_belt_speed = 0.0
        else:
            if movement == 1: self.top_belt_speed += speed_change # up
            elif movement == 2: self.top_belt_speed -= speed_change # down

        # Bottom belt control
        if shift_held:
            self.bottom_belt_speed = 0.0
        else:
            if movement == 4: self.bottom_belt_speed += speed_change # right
            elif movement == 3: self.bottom_belt_speed -= speed_change # left

        # Clamp speeds
        self.top_belt_speed = max(0.0, min(1.0, self.top_belt_speed))
        self.bottom_belt_speed = max(0.0, min(1.0, self.bottom_belt_speed))

    def _update_game_state(self):
        step_reward = 0
        
        # Update belt scroll for visual effect
        self.belt_scroll_top = (self.belt_scroll_top + self.top_belt_speed * self.MAX_PIXELS_PER_STEP) % 40
        self.belt_scroll_bottom = (self.belt_scroll_bottom + self.bottom_belt_speed * self.MAX_PIXELS_PER_STEP) % 40

        # Update items
        for item in self.items:
            if item['sorted']:
                continue

            if item['belt_idx'] == 0: # On top belt
                item['pos'].x += self.top_belt_speed * self.MAX_PIXELS_PER_STEP
                if item['pos'].x > self.BELT_TRANSFER_X:
                    item['belt_idx'] = 1
                    item['pos'].y = self.BELT_Y_BOTTOM
            else: # On bottom belt
                item['pos'].x += self.bottom_belt_speed * self.MAX_PIXELS_PER_STEP

                # Check for sorting
                for bin_info in self.bins:
                    if bin_info['rect'].collidepoint(item['pos']):
                        if item['color'] == bin_info['color']:
                            # Correct sort
                            # SFX: Positive chime
                            self.score += 1
                            item['sorted'] = True
                            step_reward += 0.1
                            self._create_particles(item['pos'], item['color'], 20)
                        else:
                            # Incorrect sort
                            # SFX: Error buzz
                            self.game_over = True
                            step_reward -= 10.0
                            self._create_particles(item['pos'], (255, 255, 255), 30)
                        break
                
                # Check for falling off
                if item['pos'].x > self.BELT_END_X:
                    # SFX: Item falling sound
                    self.game_over = True
                    step_reward -= 10.0
                    self._create_particles(item['pos'], (100, 100, 100), 30)

        # Update particles
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['size'] = max(0, p['size'] - 0.1)

        return step_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        accuracy = (self.score / self.TOTAL_ITEMS) * 100 if self.TOTAL_ITEMS > 0 else 0
        return {
            "score": self.score,
            "steps": self.steps,
            "accuracy": accuracy,
            "top_belt_speed": self.top_belt_speed,
            "bottom_belt_speed": self.bottom_belt_speed
        }

    def _render_game(self):
        self._render_belts()
        self._render_bins()
        self._render_items()
        self._render_particles()

    def _render_belts(self):
        # Top belt
        pygame.draw.rect(self.screen, self.COLOR_BELT_SHADOW, (0, self.BELT_Y_TOP + self.BELT_HEIGHT - 5, self.SCREEN_WIDTH, 10))
        pygame.draw.rect(self.screen, self.COLOR_BELT, (0, self.BELT_Y_TOP, self.SCREEN_WIDTH, self.BELT_HEIGHT))
        for i in range(self.SCREEN_WIDTH // 40 + 2):
            x = (i * 40 - self.belt_scroll_top)
            pygame.draw.line(self.screen, self.COLOR_BELT_SHADOW, (x, self.BELT_Y_TOP), (x, self.BELT_Y_TOP + self.BELT_HEIGHT), 2)
        
        # Bottom belt
        pygame.draw.rect(self.screen, self.COLOR_BELT_SHADOW, (0, self.BELT_Y_BOTTOM + self.BELT_HEIGHT - 5, self.SCREEN_WIDTH, 10))
        pygame.draw.rect(self.screen, self.COLOR_BELT, (0, self.BELT_Y_BOTTOM, self.SCREEN_WIDTH, self.BELT_HEIGHT))
        for i in range(self.SCREEN_WIDTH // 40 + 2):
            x = (i * 40 - self.belt_scroll_bottom)
            pygame.draw.line(self.screen, self.COLOR_BELT_SHADOW, (x, self.BELT_Y_BOTTOM), (x, self.BELT_Y_BOTTOM + self.BELT_HEIGHT), 2)

    def _render_bins(self):
        for bin_info in self.bins:
            pygame.draw.rect(self.screen, bin_info['color'], bin_info['rect'])
            pygame.draw.rect(self.screen, tuple(c*0.5 for c in bin_info['color']), bin_info['rect'], 4)

    def _render_items(self):
        for item in reversed(self.items):
            if not item['sorted']:
                pos = (int(item['pos'].x), int(item['pos'].y))
                shadow_pos = (pos[0], pos[1] + 4)
                # Draw shadow
                pygame.gfxdraw.filled_circle(self.screen, shadow_pos[0], shadow_pos[1], self.ITEM_RADIUS, (0, 0, 0, 100))
                # Draw main circle
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ITEM_RADIUS, item['color'])
                # Draw highlight
                highlight_pos = (pos[0] - 4, pos[1] - 4)
                pygame.gfxdraw.filled_circle(self.screen, highlight_pos[0], highlight_pos[1], self.ITEM_RADIUS // 3, (255, 255, 255, 120))


    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            size = int(p['size'])
            if size > 0:
                pygame.draw.circle(self.screen, p['color'], pos, size)

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SORTED: {self.score}/{self.TOTAL_ITEMS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Accuracy display
        accuracy = (self.score / self.TOTAL_ITEMS) * 100 if self.TOTAL_ITEMS > 0 else 0
        acc_text = self.font_main.render(f"ACCURACY: {accuracy:.0f}%", True, self.COLOR_UI_TEXT)
        self.screen.blit(acc_text, (self.SCREEN_WIDTH - acc_text.get_width() - 10, 10))

        # Speed bars
        bar_width = 200
        bar_height = 20
        # Top bar
        top_bar_y = self.SCREEN_HEIGHT - 50
        pygame.draw.rect(self.screen, self.COLOR_SPEED_BAR_BG, (10, top_bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SPEED_BAR_FILL, (10, top_bar_y, int(bar_width * self.top_belt_speed), bar_height))
        top_label = self.font_small.render("TOP BELT (UP/DOWN)", True, self.COLOR_UI_TEXT)
        self.screen.blit(top_label, (10, top_bar_y - 18))

        # Bottom bar
        bot_bar_y = self.SCREEN_HEIGHT - 50
        bot_bar_x = self.SCREEN_WIDTH - bar_width - 10
        pygame.draw.rect(self.screen, self.COLOR_SPEED_BAR_BG, (bot_bar_x, bot_bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_SPEED_BAR_FILL, (bot_bar_x, bot_bar_y, int(bar_width * self.bottom_belt_speed), bar_height))
        bot_label = self.font_small.render("BOTTOM BELT (LEFT/RIGHT)", True, self.COLOR_UI_TEXT)
        self.screen.blit(bot_label, (bot_bar_x, bot_bar_y - 18))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': random.randint(20, 40),
                'color': color,
                'size': random.uniform(2, 6)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    
    # For human play
    # This part requires a display. If you are running headlessly, this will not work.
    try:
        pygame.display.set_caption("Conveyor Belt Sort")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        
        running = True
        terminated = False
        
        while running:
            if terminated:
                print(f"Episode finished. Final Score: {info['score']}. Resetting.")
                obs, info = env.reset()
                terminated = False

            # --- Human Controls ---
            movement = 0 # none
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
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False

            # --- Gym Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            # --- Rendering ---
            # The observation is already a rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit to 30 FPS for human play
    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("Cannot run __main__ block in a headless environment without a display.")
        print("The environment itself is configured for headless operation.")


    env.close()