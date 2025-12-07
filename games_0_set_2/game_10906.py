import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:16:46.442561
# Source Brief: brief_00906.md
# Brief Index: 906
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: A puzzle game where the player redirects light beams to their targets.

    The player controls a cursor on a grid and can place one of three types of blocks:
    - REFLECT: Bounces beams off its surface.
    - ABSORB: Stops beams.
    - DEFLECT: Turns beams 90 degrees.

    The goal is to have all 5 colored beams hit their matching colored targets
    within a 100-move limit. Each block placement counts as one move.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A puzzle game where the player places blocks to redirect colored light beams to their matching targets."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press space to place a block. "
        "Press shift to cycle through block types."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    CELL_SIZE = 40
    MAX_STEPS = 100
    NUM_BEAMS = 5

    # --- Colors (Dark, Neon Theme) ---
    COLOR_BG = (26, 26, 46)
    COLOR_GRID = (42, 42, 78)
    COLOR_CURSOR = (255, 255, 102, 150)
    COLOR_TEXT = (240, 240, 240)
    BLOCK_TYPES = {
        'REFLECT': 1,
        'ABSORB': 2,
        'DEFLECT': 3,
    }
    BLOCK_NAMES = {v: k for k, v in BLOCK_TYPES.items()}
    BLOCK_COLORS = {
        BLOCK_TYPES['REFLECT']: (240, 240, 240),
        BLOCK_TYPES['ABSORB']: (10, 10, 10),
        BLOCK_TYPES['DEFLECT']: (128, 128, 128),
    }
    BEAM_COLORS = [
        (255, 51, 102),   # Red
        (102, 255, 102),  # Green
        (51, 153, 255),   # Blue
        (255, 255, 102),  # Yellow
        (204, 102, 255),  # Purple
    ]

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        self.grid = None
        self.beams = []
        self.targets = []
        self.particles = []
        self.cursor_pos = None
        self.selected_block_type = None
        self.last_action = None
        self.steps = 0
        self.cumulative_reward = 0
        self.game_over = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.cumulative_reward = 0
        self.game_over = False
        
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_block_type = self.BLOCK_TYPES['REFLECT']
        self.last_action = np.array([0, 0, 0])
        self.particles = []

        self._procedurally_generate_level()
        self._recalculate_all_beams()
        
        return self._get_observation(), self._get_info()

    def _procedurally_generate_level(self):
        self.beams = []
        self.targets = []
        
        possible_coords = [(x, y) for x in range(self.GRID_COLS) for y in range(self.GRID_ROWS)]
        # Use self.np_random which is the seeded random number generator from Gymnasium
        self.np_random.shuffle(possible_coords)
        
        for i in range(self.NUM_BEAMS):
            target_pos = possible_coords.pop()
            origin_pos = possible_coords.pop()
            
            center = np.array([self.GRID_COLS / 2, self.GRID_ROWS / 2])
            origin_vec = np.array(origin_pos)
            direction_vec = center - origin_vec
            
            if abs(direction_vec[0]) > abs(direction_vec[1]):
                direction = [int(np.sign(direction_vec[0])), 0]
            else:
                direction = [0, int(np.sign(direction_vec[1]))]
            
            if direction == [0,0]: direction = [1,0]

            self.beams.append({
                'id': i, 'origin': list(origin_pos), 'direction': direction,
                'color': self.BEAM_COLORS[i], 'path': [], 'active': True
            })
            self.targets.append({
                'id': i, 'pos': list(target_pos), 'color': self.BEAM_COLORS[i], 'hit': False
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        last_space, last_shift = self.last_action[1] == 1, self.last_action[2] == 1
        
        if movement == 1: self.cursor_pos[1] -= 1
        if movement == 2: self.cursor_pos[1] += 1
        if movement == 3: self.cursor_pos[0] -= 1
        if movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)
        
        if shift_held and not last_shift:
            # SFX: UI_Cycle
            self.selected_block_type += 1
            if self.selected_block_type > max(self.BLOCK_TYPES.values()):
                self.selected_block_type = min(self.BLOCK_TYPES.values())

        if space_held and not last_space:
            cx, cy = self.cursor_pos
            if self.grid[cy, cx] == 0:
                # SFX: Block_Place
                self.steps += 1
                
                old_distances = [self._get_beam_dist_to_target(i) for i in range(self.NUM_BEAMS)]
                old_hits = sum(t['hit'] for t in self.targets)
                
                self.grid[cy, cx] = self.selected_block_type
                self._recalculate_all_beams()
                
                new_distances = [self._get_beam_dist_to_target(i) for i in range(self.NUM_BEAMS)]
                new_hits = sum(t['hit'] for t in self.targets)
                
                distance_reward = sum(old - new for old, new in zip(old_distances, new_distances))
                hit_reward = (new_hits - old_hits) * 10
                reward = distance_reward + hit_reward

        self.last_action = action
        self.cumulative_reward += reward
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if all(t['hit'] for t in self.targets):
                # SFX: Win_Jingle
                reward += 100
            else:
                # SFX: Lose_Sound
                reward -= 100
            self.cumulative_reward += reward
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_beam_dist_to_target(self, beam_id):
        beam = self.beams[beam_id]
        target = self.targets[beam_id]
        if not beam['active'] or not beam['path']:
            return self.GRID_COLS + self.GRID_ROWS
        
        endpoint = beam['path'][-1]
        return abs(endpoint[0] - target['pos'][0]) + abs(endpoint[1] - target['pos'][1])

    def _recalculate_all_beams(self):
        for target in self.targets: target['hit'] = False
        for i in range(len(self.beams)): self._trace_beam(i)
            
    def _trace_beam(self, beam_id):
        beam = self.beams[beam_id]
        beam['path'] = [beam['origin']]
        beam['active'] = True
        
        pos = np.array(beam['origin'], dtype=int)
        direction = np.array(beam['direction'], dtype=int)
        
        for _ in range(self.GRID_COLS + self.GRID_ROWS + 5):
            prev_pos = pos.copy()
            pos += direction
            beam['path'].append(list(pos))

            if not (0 <= pos[0] < self.GRID_COLS and 0 <= pos[1] < self.GRID_ROWS):
                beam['active'] = False
                self._create_particles(self._grid_to_pixel(pos, center=True), beam['color'], 5)
                break
            
            for target in self.targets:
                if target['id'] == beam['id'] and target['pos'] == list(pos):
                    target['hit'] = True
                    # SFX: Target_Hit
                    self._create_particles(self._grid_to_pixel(pos, center=True), beam['color'], 10)

            block_type = self.grid[pos[1], pos[0]]
            if block_type != 0:
                self._create_particles(self._grid_to_pixel(pos, center=True), beam['color'], 8)
                if block_type == self.BLOCK_TYPES['ABSORB']:
                    # SFX: Beam_Absorb
                    beam['active'] = False
                    break
                elif block_type == self.BLOCK_TYPES['REFLECT']:
                    # SFX: Beam_Reflect
                    if prev_pos[0] != pos[0]: direction[0] *= -1
                    if prev_pos[1] != pos[1]: direction[1] *= -1
                elif block_type == self.BLOCK_TYPES['DEFLECT']:
                    # SFX: Beam_Deflect
                    direction = np.array([-direction[1], -direction[0]], dtype=int)

    def _check_termination(self):
        win = all(t['hit'] for t in self.targets)
        loss = self.steps >= self.MAX_STEPS
        return win or loss

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.cumulative_reward, "steps": self.steps, "targets_hit": sum(t['hit'] for t in self.targets)}

    def _grid_to_pixel(self, grid_pos, center=False):
        px = int(grid_pos[0] * self.CELL_SIZE)
        py = int(grid_pos[1] * self.CELL_SIZE)
        if center:
            px += self.CELL_SIZE // 2
            py += self.CELL_SIZE // 2
        return (px, py)

    def _render_game(self):
        self._render_grid()
        self._render_targets()
        self._render_blocks()
        self._render_beams()
        self._render_particles()
        self._render_cursor()

    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_targets(self):
        for target in self.targets:
            px, py = self._grid_to_pixel(target['pos'], center=True)
            radius = self.CELL_SIZE // 3
            if target['hit']:
                glow_radius = int(radius * 1.5 + abs(math.sin(pygame.time.get_ticks() * 0.005)) * 3)
                s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*target['color'], 80), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(s, (px - glow_radius, py - glow_radius))

            pygame.gfxdraw.aacircle(self.screen, px, py, radius, target['color'])
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, target['color'])
            pygame.gfxdraw.aacircle(self.screen, px, py, radius - 2, self.COLOR_BG)
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius - 2, self.COLOR_BG)
            pygame.gfxdraw.aacircle(self.screen, px, py, radius - 4, target['color'])

    def _render_blocks(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                block_type = self.grid[r, c]
                if block_type == 0: continue
                px, py = self._grid_to_pixel((c, r))
                rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
                color = self.BLOCK_COLORS[block_type]
                
                if block_type == self.BLOCK_TYPES['ABSORB']:
                    pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 2)
                elif block_type == self.BLOCK_TYPES['REFLECT']:
                    pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
                elif block_type == self.BLOCK_TYPES['DEFLECT']:
                    pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
                    pygame.draw.aaline(self.screen, (200,200,200), rect.topleft, rect.bottomright, 2)

    def _render_beams(self):
        for beam in self.beams:
            if len(beam['path']) < 2: continue
            pixel_path = [self._grid_to_pixel(p, center=True) for p in beam['path']]
            
            color = beam['color']
            glow_color = (*color, 100)
            
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            if len(pixel_path) > 1:
                pygame.draw.lines(s, glow_color, False, pixel_path, width=7)
                pygame.draw.lines(s, (255,255,255, 180), False, pixel_path, width=3)
                pygame.draw.lines(s, color, False, pixel_path, width=1)
            self.screen.blit(s, (0,0))
            
    def _render_cursor(self):
        px, py = self._grid_to_pixel(self.cursor_pos)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, (px, py))
        pygame.draw.rect(self.screen, (255,255,255), (px, py, self.CELL_SIZE, self.CELL_SIZE), 1)

    def _render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] = max(0, p['start_radius'] * (p['life'] / p['start_life']))
            
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                color_with_alpha = (*p['color'], int(200 * p['life']/p['start_life']))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color_with_alpha)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            life = self.np_random.integers(15, 30)
            start_radius = self.np_random.uniform(2, 5)
            self.particles.append({
                'pos': list(pos), 'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color, 'life': life, 'start_life': life,
                'radius': start_radius, 'start_radius': start_radius
            })

    def _render_ui(self):
        moves_text = f"MOVES: {self.MAX_STEPS - self.steps}/{self.MAX_STEPS}"
        text_surf = self.font_large.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))
        
        block_name = self.BLOCK_NAMES[self.selected_block_type]
        block_text = f"SELECTED: {block_name}"
        text_surf = self.font_small.render(block_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(bottomright=(self.SCREEN_WIDTH - 10, self.SCREEN_HEIGHT - 10))
        self.screen.blit(text_surf, text_rect)
        
        preview_rect = pygame.Rect(0, 0, 20, 20)
        preview_rect.right = text_rect.left - 10
        preview_rect.centery = text_rect.centery
        
        pygame.draw.rect(self.screen, self.BLOCK_COLORS[self.selected_block_type], preview_rect)
        if self.selected_block_type == self.BLOCK_TYPES['DEFLECT']:
            pygame.draw.aaline(self.screen, (200,200,200), preview_rect.topleft, preview_rect.bottomright)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # Un-comment the line below to run with a display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Beam Redirect")
    
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        keys = pygame.key.get_pressed()
        
        current_action = [0, 0, 0]
        if keys[pygame.K_UP]: current_action[0] = 1
        elif keys[pygame.K_DOWN]: current_action[0] = 2
        elif keys[pygame.K_LEFT]: current_action[0] = 3
        elif keys[pygame.K_RIGHT]: current_action[0] = 4
        
        if keys[pygame.K_SPACE]: current_action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: current_action[2] = 1

        obs, reward, terminated, truncated, info = env.step(np.array(current_action))
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(30)
        
    env.close()