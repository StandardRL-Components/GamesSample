import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


# Set the SDL video driver to dummy to run Pygame headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Stack colored blocks to build a tower. Earn points by matching colors consecutively and avoid penalties for mismatches."
    user_guide = "Controls: ← to place a block on the left, → for the right, and ↓ for the center."
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_HEIGHT = 50
    BLOCK_WIDTH = 50
    BLOCK_HEIGHT = 15
    MAX_TOWER_HEIGHT = 50
    WIN_SCORE = 75
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG_TOP = (15, 20, 40)
    COLOR_BG_BOTTOM = (40, 50, 80)
    COLOR_GROUND = (25, 30, 55)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 20)
    COLOR_UI_BAR = (80, 100, 150)
    COLOR_UI_BAR_FILL = (150, 180, 255)
    
    BLOCK_COLORS = {
        0: {"name": "Red", "val": (255, 80, 80), "dark": (180, 50, 50)},
        1: {"name": "Green", "val": (80, 255, 80), "dark": (50, 180, 50)},
        2: {"name": "Blue", "val": (80, 150, 255), "dark": (50, 100, 180)},
        3: {"name": "Yellow", "val": (255, 220, 80), "dark": (180, 150, 50)},
    }
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        try:
            self.font_big = pygame.font.SysFont("Consolas", 36, bold=True)
            self.font_medium = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 16)
        except pygame.error:
            self.font_big = pygame.font.SysFont(None, 48)
            self.font_medium = pygame.font.SysFont(None, 32)
            self.font_small = pygame.font.SysFont(None, 22)
        
        self.tower_blocks = []
        self.pending_block_color_id = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_result_text = ""
        self.last_color_id = -1
        self.consecutive_matches = 0
        self.particles = []
        self.glows = []
        self.screen_shake = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_result_text = ""
        
        self.tower_blocks = []
        self.last_color_id = -1
        self.consecutive_matches = 0
        
        self.particles = []
        self.glows = []
        self.screen_shake = 0
        
        self.pending_block_color_id = self.np_random.integers(0, len(self.BLOCK_COLORS))
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        truncated = False

        # --- Action to Column Mapping ---
        if movement == 3:  # Left
            column = 0
        elif movement == 4:  # Right
            column = 2
        else:  # None, Up, Down (maps to center)
            column = 1
        
        # --- Block Placement Logic ---
        highest_y = self.SCREEN_HEIGHT - self.GROUND_HEIGHT
        block_below = None
        for block in self.tower_blocks:
            if block['column'] == column and block['rect'].top < highest_y:
                highest_y = block['rect'].top
                block_below = block

        new_block_pos_x = (self.SCREEN_WIDTH // 2 - self.BLOCK_WIDTH // 2) + (column - 1) * (self.BLOCK_WIDTH + 5)
        new_block_pos_y = highest_y - self.BLOCK_HEIGHT
        
        new_block_rect = pygame.Rect(new_block_pos_x, new_block_pos_y, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
        new_block_color_id = self.pending_block_color_id
        
        # --- Reward Calculation ---
        if block_below is None: # First block in a column
            reward = 0.1 # Small reward for starting a stack
            self.consecutive_matches = 0
        else:
            if block_below['color_id'] == new_block_color_id: # Color Match
                reward = 1.0
                self.consecutive_matches += 1
                if self.consecutive_matches > 1:
                    bonus = min(5.0 * (self.consecutive_matches - 1), 25) # Capped bonus
                    reward += bonus
                self._create_particles(new_block_rect.center, self.BLOCK_COLORS[new_block_color_id]['val'])
                self._create_glow(new_block_rect.center, self.BLOCK_COLORS[new_block_color_id]['val'])
            else: # Mismatch
                penalty = 1.0 + (len(self.tower_blocks) // 10) * 0.5
                reward = -penalty
                self.consecutive_matches = 0
                self.screen_shake = 8

        self.score += reward
        self.last_color_id = new_block_color_id

        self.tower_blocks.append({
            'rect': new_block_rect,
            'color_id': new_block_color_id,
            'column': column,
            'id': len(self.tower_blocks)
        })

        # --- Prepare next block ---
        self.pending_block_color_id = self.np_random.integers(0, len(self.BLOCK_COLORS))

        self.steps += 1
        
        # --- Termination Check ---
        tower_height = len(self.tower_blocks)
        if self.score < -10: # Allow for some negative score before ending
            terminated = True
            reward -= 25 # Terminal penalty for going bankrupt
            self.game_result_text = "BANKRUPT!"
        elif tower_height >= self.MAX_TOWER_HEIGHT:
            terminated = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
                self.game_result_text = "TOWER COMPLETE!"
            else:
                reward -= 100 # Lose penalty
                self.game_result_text = "GOAL NOT MET!"
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_result_text = "TIME UP!"
        
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        # --- Background Gradient ---
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

        # --- Screen Shake ---
        render_offset = [0, 0]
        if self.screen_shake > 0:
            self.screen_shake -= 1
            render_offset[0] = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)
            render_offset[1] = self.np_random.integers(-self.screen_shake, self.screen_shake + 1)

        # --- Update and Render Effects ---
        self._update_and_draw_glows(render_offset)
        self._update_and_draw_particles(render_offset)

        # --- Render Ground ---
        ground_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.GROUND_HEIGHT, self.SCREEN_WIDTH, self.GROUND_HEIGHT)
        ground_rect.move_ip(render_offset)
        pygame.draw.rect(self.screen, self.COLOR_GROUND, ground_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_BAR, ground_rect.topleft, ground_rect.topright, 2)

        # --- Render Tower Blocks ---
        wobble_amplitude = min(2.5, len(self.tower_blocks) / 20.0)
        for block in self.tower_blocks:
            wobble_offset = math.sin(self.steps * 0.15 + block['id'] * 0.5) * wobble_amplitude
            block_rect = block['rect'].copy()
            block_rect.x += wobble_offset
            block_rect.move_ip(render_offset)
            
            color_data = self.BLOCK_COLORS[block['color_id']]
            self._draw_block(self.screen, block_rect, color_data['val'], color_data['dark'])

        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # --- Score Display ---
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, self.font_medium, (20, 20))
        
        # --- Height Indicator Bar ---
        bar_width = 20
        bar_height = 150
        bar_x = self.SCREEN_WIDTH - bar_width - 20
        bar_y = 20
        
        height_ratio = min(1.0, len(self.tower_blocks) / self.MAX_TOWER_HEIGHT)
        
        frame_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, frame_rect, 2, border_radius=4)
        
        fill_height = bar_height * height_ratio
        fill_rect = pygame.Rect(bar_x, bar_y + bar_height - fill_height, bar_width, fill_height)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, fill_rect, 0, border_radius=4)

        self._draw_text(f"{len(self.tower_blocks)}/{self.MAX_TOWER_HEIGHT}", self.font_small, (bar_x + bar_width/2, bar_y + bar_height + 10), center=True)
        
        # --- Next Block Preview ---
        self._draw_text("NEXT:", self.font_small, (self.SCREEN_WIDTH / 2, 20), center=True)
        preview_rect = pygame.Rect(self.SCREEN_WIDTH/2 - self.BLOCK_WIDTH/2, 40, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
        color_data = self.BLOCK_COLORS[self.pending_block_color_id]
        self._draw_block(self.screen, preview_rect, color_data['val'], color_data['dark'])

        # --- Game Over Text ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            self._draw_text(self.game_result_text, self.font_big, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20), center=True)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "tower_height": len(self.tower_blocks),
            "consecutive_matches": self.consecutive_matches,
        }
        
    def _draw_block(self, surface, rect, main_color, dark_color):
        pygame.draw.rect(surface, dark_color, rect, border_radius=3)
        inner_rect = rect.copy()
        inner_rect.height -= 3
        pygame.draw.rect(surface, main_color, inner_rect, border_radius=3)

    def _draw_text(self, text, font, pos, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
            
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'life': random.randint(20, 40),
                'color': color,
                'size': random.uniform(2, 5)
            })
    
    def _update_and_draw_particles(self, offset):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            p['size'] -= 0.05
            
            if p['life'] <= 0 or p['size'] <= 0:
                self.particles.remove(p)
                continue
            
            pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            pygame.draw.circle(self.screen, p['color'], pos, int(p['size']))

    def _create_glow(self, pos, color, radius=40, life=20):
        self.glows.append({'pos': pos, 'color': color, 'radius': 5, 'max_radius': radius, 'life': life, 'max_life': life})

    def _update_and_draw_glows(self, offset):
        for g in self.glows[:]:
            g['life'] -= 1
            if g['life'] <= 0:
                self.glows.remove(g)
                continue
            
            life_ratio = g['life'] / g['max_life']
            current_radius = int(g['max_radius'] * (1 - life_ratio**2))
            alpha = int(150 * life_ratio)
            
            pos = (int(g['pos'][0] + offset[0]), int(g['pos'][1] + offset[1]))
            
            temp_surf = pygame.Surface((current_radius * 2, current_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*g['color'], alpha), (current_radius, current_radius), current_radius)
            self.screen.blit(temp_surf, (pos[0] - current_radius, pos[1] - current_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example of how to run the environment ---
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Un-set the dummy driver to allow for display
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Tower Stacker")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action_taken = False
        action = np.array([0, 0, 0]) # Default action: place center

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                    truncated = False
                    action_taken = True

                if not terminated and not truncated:
                    if event.key == pygame.K_LEFT:
                        action = np.array([3, 0, 0])
                        action_taken = True
                    elif event.key == pygame.K_RIGHT:
                        action = np.array([4, 0, 0])
                        action_taken = True
                    elif event.key == pygame.K_DOWN:
                        action = np.array([0, 0, 0]) # Center
                        action_taken = True
        
        if action_taken:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action[0]}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Height: {info['tower_height']}, Terminated: {terminated}, Truncated: {truncated}")

        # Draw the observation to the screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS

    env.close()