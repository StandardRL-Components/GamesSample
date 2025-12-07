import gymnasium as gym
import os
import pygame
import numpy as np
import pygame.gfxdraw
import math
import random
from gymnasium.spaces import MultiDiscrete
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" to run Pygame in a headless environment.
# This is essential for compatibility with test servers and cloud environments.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player synchronizes dual conveyor belts
    to match colored blocks to target patterns within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # FIX: Add game description
    game_description = (
        "Synchronize dual conveyor belts to match colored blocks to target patterns before time runs out."
    )

    # FIX: Add user guide
    user_guide = (
        "Use ↑/↓ to adjust the top belt's speed and ←/→ for the bottom belt's speed. Match the moving blocks to the target pattern in the center."
    )

    # FIX: Add auto_advance flag
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # --- Visual & Game Parameters ---
        self._define_colors_and_fonts()
        self._define_game_parameters()
        self._generate_patterns()

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = "PLAYING"
        self.belt_speeds = [0.0, 0.0]
        self.time_remaining = 0.0
        self.current_pattern_index = 0
        self.belt1_blocks = []
        self.belt2_blocks = []
        self.particles = []
        self.match_feedback_state = {'status': 'IDLE', 'timer': 0}

    def _define_colors_and_fonts(self):
        """Define all visual assets like colors and fonts."""
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_BELT = (50, 60, 70)
        self.COLOR_BELT_SHADOW = (30, 40, 50)
        self.COLOR_UI_FRAME = (100, 110, 120)
        self.COLOR_UI_BG = (40, 50, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SUCCESS = (100, 255, 100)
        self.COLOR_FAIL = (255, 100, 100)
        self.COLOR_HINT = (100, 150, 255)

        self.BLOCK_COLORS = {
            "RED": (255, 80, 80),
            "GREEN": (80, 255, 80),
            "BLUE": (80, 120, 255),
            "YELLOW": (255, 255, 80),
            "PURPLE": (200, 80, 255),
        }
        self.COLOR_LIST = list(self.BLOCK_COLORS.values())
        self.FONT_UI = pygame.font.SysFont("Consolas", 20, bold=True)
        self.FONT_BIG = pygame.font.SysFont("Consolas", 60, bold=True)

    def _define_game_parameters(self):
        """Define constants for game mechanics and layout."""
        self.MAX_TIME = 90.0
        self.MAX_STEPS = int(self.MAX_TIME * self.metadata["render_fps"])
        self.NUM_PATTERNS = 10
        self.BELT_WIDTH = self.screen_height * 0.2
        self.BELT_SPACING = self.screen_height * 0.25
        self.BELT_Y_1 = self.screen_height / 2 - self.BELT_SPACING
        self.BELT_Y_2 = self.screen_height / 2 + self.BELT_SPACING
        self.BLOCK_SIZE = self.BELT_WIDTH * 0.8
        self.BLOCK_SPACING = self.BLOCK_SIZE * 2.5
        self.MAX_BELT_SPEED = 5.0  # pixels per step
        self.NUM_BLOCKS_PER_BELT = 20

        self.TARGET_ZONE_WIDTH = 300
        self.TARGET_ZONE_X = (self.screen_width - self.TARGET_ZONE_WIDTH) / 2

    def _generate_patterns(self):
        """Generate a list of target patterns for the game."""
        self.patterns = []
        color_keys = list(self.BLOCK_COLORS.values()) + [None]
        for i in range(self.NUM_PATTERNS):
            pattern_length = min(4, i // 2 + 1)
            pattern = []
            for _ in range(pattern_length):
                c1 = random.choice(color_keys)
                c2 = random.choice(color_keys)
                if c1 is None and c2 is None: # Avoid fully empty slots
                    c1 = random.choice(self.COLOR_LIST)
                pattern.append((c1, c2))
            self.patterns.append(pattern)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_status = "PLAYING"
        self.belt_speeds = [0.5, 0.5]  # Initial speed as percentage
        self.time_remaining = self.MAX_TIME
        self.current_pattern_index = 0
        self.particles = []
        self.match_feedback_state = {'status': 'IDLE', 'timer': 0}

        self.belt1_blocks = self._create_belt_blocks(self.NUM_BLOCKS_PER_BELT)
        self.belt2_blocks = self._create_belt_blocks(self.NUM_BLOCKS_PER_BELT)

        return self._get_observation(), self._get_info()

    def _create_belt_blocks(self, num_blocks):
        blocks = []
        for i in range(num_blocks):
            blocks.append({
                'pos': -i * self.BLOCK_SPACING,
                'color': self.np_random.choice(self.COLOR_LIST, axis=0)
            })
        return blocks

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()

        reward, terminated = self._process_game_logic()
        
        self.score += reward

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.win_status = "LOSE"

        self.game_over = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        speed_change = 0.02

        if movement == 1: self.belt_speeds[0] += speed_change # Up
        elif movement == 2: self.belt_speeds[0] -= speed_change # Down
        elif movement == 4: self.belt_speeds[1] += speed_change # Right
        elif movement == 3: self.belt_speeds[1] -= speed_change # Left

        self.belt_speeds[0] = np.clip(self.belt_speeds[0], 0, 1)
        self.belt_speeds[1] = np.clip(self.belt_speeds[1], 0, 1)

    def _update_game_state(self):
        self.steps += 1
        self.time_remaining -= 1.0 / self.metadata["render_fps"]

        # Update block positions
        for block in self.belt1_blocks:
            block['pos'] += self.belt_speeds[0] * self.MAX_BELT_SPEED
        for block in self.belt2_blocks:
            block['pos'] += self.belt_speeds[1] * self.MAX_BELT_SPEED

        # Recycle blocks
        self._recycle_blocks(self.belt1_blocks)
        self._recycle_blocks(self.belt2_blocks)

        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1

        # Update feedback timer
        if self.match_feedback_state['timer'] > 0:
            self.match_feedback_state['timer'] -= 1

    def _recycle_blocks(self, blocks):
        farthest_pos = min(b['pos'] for b in blocks) if blocks else 0
        for block in blocks:
            if block['pos'] > self.screen_width + self.BLOCK_SIZE:
                block['pos'] = farthest_pos - self.BLOCK_SPACING
                block['color'] = self.np_random.choice(self.COLOR_LIST, axis=0)

    def _process_game_logic(self):
        reward = 0.0
        
        current_pattern = self.patterns[self.current_pattern_index]
        pattern_len = len(current_pattern)
        slot_width = self.TARGET_ZONE_WIDTH / pattern_len
        
        is_perfect_match = True
        is_mismatch = False

        for i in range(pattern_len):
            slot_x_start = self.TARGET_ZONE_X + i * slot_width
            slot_x_end = slot_x_start + slot_width
            target_c1, target_c2 = current_pattern[i]

            block1, in_slot1 = self._get_block_in_slot(self.belt1_blocks, slot_x_start, slot_x_end)
            if target_c1 is None:
                if in_slot1: reward -= 0.01; is_perfect_match = False
            else:
                if not in_slot1: is_perfect_match = False
                # FIX: Use np.array_equal for comparing numpy array (block color) and tuple (target color)
                elif np.array_equal(block1['color'], target_c1): reward += 0.1
                else: reward -= 0.01; is_perfect_match = False; is_mismatch = True

            block2, in_slot2 = self._get_block_in_slot(self.belt2_blocks, slot_x_start, slot_x_end)
            if target_c2 is None:
                if in_slot2: reward -= 0.01; is_perfect_match = False
            else:
                if not in_slot2: is_perfect_match = False
                # FIX: Use np.array_equal for comparing numpy array (block color) and tuple (target color)
                elif np.array_equal(block2['color'], target_c2): reward += 0.1
                else: reward -= 0.01; is_perfect_match = False; is_mismatch = True
        
        if is_perfect_match:
            reward += 10.0
            self.current_pattern_index += 1
            self.match_feedback_state = {'status': 'SUCCESS', 'timer': 20}
            for _ in range(50):
                self._create_particle(self.screen_width/2, self.screen_height/2)
        elif is_mismatch:
             self.match_feedback_state = {'status': 'FAIL', 'timer': 5}
        else:
             self.match_feedback_state = {'status': 'IDLE', 'timer': 0}

        terminated = False
        if self.current_pattern_index >= self.NUM_PATTERNS:
            reward += 100.0
            terminated = True
            self.win_status = "WIN"
        elif self.time_remaining <= 0:
            reward -= 50.0
            terminated = True
            self.win_status = "LOSE"

        return reward, terminated

    def _get_block_in_slot(self, blocks, x_start, x_end):
        for block in blocks:
            center_x = block['pos'] + self.BLOCK_SIZE / 2
            if x_start < center_x < x_end:
                return block, True
        return None, False
        
    def _create_particle(self, x, y):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 5)
        self.particles.append({
            'pos': [x, y],
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'life': self.np_random.integers(20, 41),
            'color': self.COLOR_LIST[self.np_random.integers(len(self.COLOR_LIST))]
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "patterns_completed": self.current_pattern_index,
            "time_remaining": self.time_remaining,
        }

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_BELT_SHADOW, (0, self.BELT_Y_1 - self.BELT_WIDTH/2 + 5, self.screen_width, self.BELT_WIDTH))
        pygame.draw.rect(self.screen, self.COLOR_BELT, (0, self.BELT_Y_1 - self.BELT_WIDTH/2, self.screen_width, self.BELT_WIDTH))
        pygame.draw.rect(self.screen, self.COLOR_BELT_SHADOW, (0, self.BELT_Y_2 - self.BELT_WIDTH/2 + 5, self.screen_width, self.BELT_WIDTH))
        pygame.draw.rect(self.screen, self.COLOR_BELT, (0, self.BELT_Y_2 - self.BELT_WIDTH/2, self.screen_width, self.BELT_WIDTH))
        
        self._draw_cog(self.screen, (20, self.BELT_Y_1), self.BELT_WIDTH * 0.4, 8, self.belt_speeds[0] * self.steps)
        self._draw_cog(self.screen, (20, self.BELT_Y_2), self.BELT_WIDTH * 0.4, 8, self.belt_speeds[1] * self.steps)

        for block in self.belt1_blocks:
            self._draw_block(self.screen, (block['pos'], self.BELT_Y_1), self.BLOCK_SIZE, block['color'])
        for block in self.belt2_blocks:
            self._draw_block(self.screen, (block['pos'], self.BELT_Y_2), self.BLOCK_SIZE, block['color'])
            
        for p in self.particles:
            size = max(0, p['life'] / 10)
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0], p['pos'][1], size, size))

    def _render_ui(self):
        feedback_color = self.COLOR_UI_FRAME
        if self.match_feedback_state['timer'] > 0:
            if self.match_feedback_state['status'] == 'SUCCESS':
                feedback_color = self.COLOR_SUCCESS
            elif self.match_feedback_state['status'] == 'FAIL':
                feedback_color = self.COLOR_FAIL
        
        pygame.draw.rect(self.screen, feedback_color, (self.TARGET_ZONE_X - 2, 0, self.TARGET_ZONE_WIDTH + 4, self.screen_height), 2, border_radius=5)
        
        ui_bar_rect = pygame.Rect(0, 0, self.screen_width, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_bar_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_FRAME, (0, 40), (self.screen_width, 40), 2)
        
        progress_text = f"Patterns: {self.current_pattern_index} / {self.NUM_PATTERNS}"
        self._draw_text(progress_text, (10, 10), self.FONT_UI, self.COLOR_TEXT)
        
        timer_text = f"Time: {max(0, self.time_remaining):.1f}"
        timer_color = self.COLOR_TEXT if self.time_remaining > 10 else self.COLOR_FAIL
        self._draw_text(timer_text, (self.screen_width - 150, 10), self.FONT_UI, timer_color)

        if not self.game_over and self.current_pattern_index < self.NUM_PATTERNS:
            current_pattern = self.patterns[self.current_pattern_index]
            pattern_len = len(current_pattern)
            slot_width = self.TARGET_ZONE_WIDTH / pattern_len
            
            for i, (c1, c2) in enumerate(current_pattern):
                slot_x = self.TARGET_ZONE_X + i * slot_width
                
                if c1: self._draw_block(self.screen, (slot_x + (slot_width - self.BLOCK_SIZE)/2, self.BELT_Y_1), self.BLOCK_SIZE, c1, alpha=100)
                else: self._draw_block_placeholder(self.screen, (slot_x + (slot_width - self.BLOCK_SIZE)/2, self.BELT_Y_1), self.BLOCK_SIZE)
                
                if c2: self._draw_block(self.screen, (slot_x + (slot_width - self.BLOCK_SIZE)/2, self.BELT_Y_2), self.BLOCK_SIZE, c2, alpha=100)
                else: self._draw_block_placeholder(self.screen, (slot_x + (slot_width - self.BLOCK_SIZE)/2, self.BELT_Y_2), self.BLOCK_SIZE)

        if self.game_over:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "YOU WIN!" if self.win_status == "WIN" else "TIME'S UP!"
            color = self.COLOR_SUCCESS if self.win_status == "WIN" else self.COLOR_FAIL
            
            text_surf = self.FONT_BIG.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.screen_width/2, self.screen_height/2))
            self.screen.blit(text_surf, text_rect)

    def _draw_text(self, text, pos, font, color):
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

    def _draw_block(self, surface, pos, size, color, alpha=255):
        x, y_center = pos
        y = y_center - size / 2
        main_rect = pygame.Rect(int(x), int(y), int(size), int(size))
        color_tuple = tuple(int(c) for c in color)

        if alpha < 255:
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            s.fill((*color_tuple, alpha))
            shadow_color = tuple(int(c * 0.8) for c in color_tuple)
            pygame.gfxdraw.rectangle(s, s.get_rect(), (*shadow_color, alpha))
            surface.blit(s, main_rect.topleft)
        else:
            highlight = tuple(min(255, c + 40) for c in color_tuple)
            shadow = tuple(max(0, c - 40) for c in color_tuple)
            pygame.draw.rect(surface, shadow, main_rect.move(2, 2))
            pygame.draw.rect(surface, color_tuple, main_rect)
            pygame.draw.line(surface, highlight, main_rect.topleft, main_rect.topright, 2)
            pygame.draw.line(surface, highlight, main_rect.topleft, main_rect.bottomleft, 2)

    def _draw_block_placeholder(self, surface, pos, size):
        x, y_center = pos
        y = y_center - size / 2
        rect = pygame.Rect(int(x), int(y), int(size), int(size))
        pygame.gfxdraw.rectangle(surface, rect, (*self.COLOR_UI_FRAME, 100))

    def _draw_cog(self, surface, center, radius, teeth, angle_offset):
        color = self.COLOR_BELT_SHADOW
        angle_step = 360 / (teeth * 2)
        points = []
        for i in range(teeth * 2):
            angle = math.radians(i * angle_step + angle_offset)
            r = radius if i % 2 == 0 else radius * 0.7
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            points.append((int(x), int(y)))
        
        if len(points) > 2:
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius*0.5), color)
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius*0.5), color)


if __name__ == '__main__':
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        print("Cannot run __main__ with SDL_VIDEODRIVER=dummy. Set a different video driver or unset the variable to run interactively.")
        exit()

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Conveyor Belt Sync")
    done = False
    
    while not done:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: action[0] = 1
        elif keys[pygame.K_s]: action[0] = 2
        elif keys[pygame.K_d]: action[0] = 4
        elif keys[pygame.K_a]: action[0] = 3
        else: action[0] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Status: {env.win_status}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.metadata["render_fps"])

    pygame.quit()