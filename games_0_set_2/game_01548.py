
# Generated: 2025-08-28T00:03:35.634624
# Source Brief: brief_01548.md
# Brief Index: 1548

        
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
        "Controls: Use arrow keys to select a crystal. Hold Shift to rotate counter-clockwise, "
        "or Space to rotate clockwise. Illuminate all 5 targets to win."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A puzzle game where you rotate crystals to redirect a light beam. "
        "Solve the puzzle by illuminating all target crystals before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_end = pygame.font.SysFont("monospace", 48, bold=True)
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000 # Safety limit
        
        self.GRID_WIDTH, self.GRID_HEIGHT = 14, 10
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 28, 14
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 60)
        self.COLOR_CRYSTAL = (100, 150, 255)
        self.COLOR_CRYSTAL_LIT = (255, 255, 255)
        self.COLOR_CRYSTAL_SELECT = (255, 255, 0)
        self.COLOR_TARGET_OFF = (80, 0, 120)
        self.COLOR_TARGET_ON = (255, 50, 255)
        self.COLOR_BEAM = (255, 220, 50)
        self.COLOR_BEAM_GLOW = (255, 180, 0, 100) # with alpha
        self.COLOR_TEXT = (220, 220, 220)
        
        # --- Reflection Logic (Hex directions: 0:E, 1:NE, 2:NW, 3:W, 4:SW, 5:SE) ---
        # Pre-calculated reflection table for 3 mirror axes
        # self.REFLECTIONS[orientation][in_direction] = out_direction
        self.REFLECTIONS = [
            # Orientation 0 (NE-SW axis mirror)
            {0: 4, 1: 1, 2: 5, 3: 2, 4: 4, 5: 2},
            # Orientation 1 (E-W axis mirror)
            {0: 0, 1: 5, 2: 4, 3: 3, 4: 2, 5: 1},
            # Orientation 2 (NW-SE axis mirror)
            {0: 2, 1: 3, 2: 2, 3: 1, 4: 0, 5: 5},
        ]
        self.NUM_ROTATIONS = len(self.REFLECTIONS)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.light_source = {}
        self.rotatable_crystals = []
        self.targets = []
        self.selected_crystal_idx = 0
        self.beam_path = []
        self.particles = []
        self.last_lit_target_count = 0
        self.np_random = None

        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _generate_puzzle(self):
        self.light_source = {'grid_pos': (0, self.GRID_HEIGHT // 2), 'direction': 0}
        
        occupied_pos = {self.light_source['grid_pos']}
        
        self.targets = []
        for _ in range(5):
            pos = self._get_random_unoccupied_pos(occupied_pos, edge_buffer=1)
            self.targets.append({'grid_pos': pos, 'is_lit': False})
            occupied_pos.add(pos)

        self.rotatable_crystals = []
        for _ in range(4):
            pos = self._get_random_unoccupied_pos(occupied_pos, edge_buffer=1)
            self.rotatable_crystals.append({'grid_pos': pos, 'rotation': 0})
            occupied_pos.add(pos)
            
        self.rotatable_crystals.sort(key=lambda c: (c['grid_pos'][1], c['grid_pos'][0]))
        
        num_scrambles = self.np_random.integers(10, 20)
        for _ in range(num_scrambles):
            idx = self.np_random.integers(0, len(self.rotatable_crystals))
            direction = self.np_random.choice([-1, 1])
            self.rotatable_crystals[idx]['rotation'] = (self.rotatable_crystals[idx]['rotation'] + direction) % self.NUM_ROTATIONS

    def _get_random_unoccupied_pos(self, occupied, edge_buffer=0):
        while True:
            x = self.np_random.integers(edge_buffer, self.GRID_WIDTH - edge_buffer)
            y = self.np_random.integers(edge_buffer, self.GRID_HEIGHT - edge_buffer)
            if (x, y) not in occupied:
                return (x, y)

    def _calculate_beam_path(self):
        self.beam_path = []
        for t in self.targets:
            t['is_lit'] = False

        beam_pos = self.light_source['grid_pos']
        beam_dir = self.light_source['direction']
        
        self._spawn_particles(self._iso_to_screen(*beam_pos), 5)
        self.beam_path.append(self._iso_to_screen(*beam_pos))

        for _ in range(self.GRID_WIDTH + self.GRID_HEIGHT):
            next_pos = self._get_next_pos(beam_pos, beam_dir)
            
            hit_crystal = None
            for crystal in self.rotatable_crystals:
                if crystal['grid_pos'] == next_pos:
                    hit_crystal = crystal
                    break
            
            for target in self.targets:
                if target['grid_pos'] == next_pos:
                    target['is_lit'] = True
            
            beam_pos = next_pos
            self.beam_path.append(self._iso_to_screen(*beam_pos))

            if hit_crystal:
                # sfx_reflect.wav
                self._spawn_particles(self._iso_to_screen(*beam_pos), 10, self.COLOR_CRYSTAL_LIT)
                rot = hit_crystal['rotation']
                beam_dir = self.REFLECTIONS[rot][beam_dir]
            
            if not (0 <= beam_pos[0] < self.GRID_WIDTH and 0 <= beam_pos[1] < self.GRID_HEIGHT):
                break

    def _get_next_pos(self, pos, direction):
        x, y = pos
        is_odd = x % 2
        if direction == 0: return (x + 1, y)
        if direction == 1: return (x + 1, y - 1 + is_odd)
        if direction == 2: return (x - 1, y - 1 + is_odd)
        if direction == 3: return (x - 1, y)
        if direction == 4: return (x - 1, y + is_odd)
        if direction == 5: return (x + 1, y + is_odd)
        return pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        self.selected_crystal_idx = 0
        self.particles = []

        self._generate_puzzle()
        self._calculate_beam_path()
        
        self.last_lit_target_count = sum(1 for t in self.targets if t['is_lit'])

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        action_taken = False
        
        num_crystals = len(self.rotatable_crystals)
        if movement in [1, 3]: # Up or Left
            self.selected_crystal_idx = (self.selected_crystal_idx - 1 + num_crystals) % num_crystals
        elif movement in [2, 4]: # Down or Right
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % num_crystals

        if space_held:
            # sfx_rotate_cw.wav
            self.rotatable_crystals[self.selected_crystal_idx]['rotation'] = \
                (self.rotatable_crystals[self.selected_crystal_idx]['rotation'] + 1) % self.NUM_ROTATIONS
            action_taken = True
        elif shift_held:
            # sfx_rotate_ccw.wav
            self.rotatable_crystals[self.selected_crystal_idx]['rotation'] = \
                (self.rotatable_crystals[self.selected_crystal_idx]['rotation'] - 1 + self.NUM_ROTATIONS) % self.NUM_ROTATIONS
            action_taken = True

        if action_taken:
            self.moves_left -= 1
            
        self._calculate_beam_path()
        
        current_lit_count = sum(1 for t in self.targets if t['is_lit'])
        
        if action_taken:
            newly_lit = max(0, current_lit_count - self.last_lit_target_count)
            if newly_lit > 0:
                # sfx_target_lit.wav
                reward += 5.0 * newly_lit
            reward += 0.1 * (current_lit_count - self.last_lit_target_count)

        self.last_lit_target_count = current_lit_count
        self.score += reward
        
        terminated = False
        if current_lit_count == len(self.targets):
            # sfx_win.wav
            reward += 100.0
            self.score += 100.0
            terminated = True
            self.game_over = True
        elif self.moves_left <= 0:
            reward -= 50.0
            self.score -= 50.0
            terminated = True
            self.game_over = True
            
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_particles()
        self._render_cavern_floor()
        self._render_targets()
        self._render_beam()
        self._render_crystals()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "targets_lit": self.last_lit_target_count,
        }
        
    def _render_cavern_floor(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                points = [
                    self._iso_to_screen(x, y),
                    self._iso_to_screen(x + 1, y),
                    self._iso_to_screen(x + 1, y + 1),
                    self._iso_to_screen(x, y + 1),
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

    def _render_targets(self):
        for target in self.targets:
            sx, sy = self._iso_to_screen(*target['grid_pos'])
            color = self.COLOR_TARGET_ON if target['is_lit'] else self.COLOR_TARGET_OFF
            points = [(sx, sy - 8), (sx + 8, sy), (sx, sy + 8), (sx - 8, sy)]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TEXT)
            if target['is_lit']:
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, 12, (*color, 50))

    def _render_crystals(self):
        for i, crystal in enumerate(self.rotatable_crystals):
            sx, sy = self._iso_to_screen(*crystal['grid_pos'])
            is_selected = (i == self.selected_crystal_idx)
            
            if is_selected:
                pygame.gfxdraw.filled_circle(self.screen, sx, sy, 20, (*self.COLOR_CRYSTAL_SELECT, 60))
                pygame.gfxdraw.aacircle(self.screen, sx, sy, 20, self.COLOR_CRYSTAL_SELECT)

            size = 12
            points = []
            for j in range(6):
                angle = math.pi / 3 * j
                points.append((sx + size * math.cos(angle), sy + size * math.sin(angle)))
            
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL_LIT)

            rot = crystal['rotation']
            if rot == 0: # NE-SW
                p1 = (sx - size * 0.7, sy - size * 0.7)
                p2 = (sx + size * 0.7, sy + size * 0.7)
            elif rot == 1: # E-W
                p1 = (sx - size, sy)
                p2 = (sx + size, sy)
            else: # NW-SE
                p1 = (sx - size * 0.7, sy + size * 0.7)
                p2 = (sx + size * 0.7, sy - size * 0.7)
            pygame.draw.aaline(self.screen, self.COLOR_CRYSTAL_LIT, p1, p2, 2)
                
    def _render_beam(self):
        if len(self.beam_path) > 1:
            glow_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.draw.aalines(glow_surface, (*self.COLOR_BEAM, 100), False, self.beam_path, 8)
            self.screen.blit(glow_surface, (0,0))
            pygame.draw.aalines(self.screen, self.COLOR_BEAM, False, self.beam_path, 2)
            
    def _update_and_draw_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            size = max(1, int(p['life'] / p['max_life'] * 4))
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _spawn_particles(self, pos, count, color=None):
        if color is None: color = self.COLOR_BEAM
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 2.0)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life, 'max_life': life, 'color': color
            })
            
    def _render_ui(self):
        moves_text = f"MOVES: {self.moves_left}/{self.MAX_MOVES}"
        text_surf = self.font_ui.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "PUZZLE SOLVED" if self.last_lit_target_count == len(self.targets) else "OUT OF MOVES"
            color = self.COLOR_CRYSTAL_SELECT if self.last_lit_target_count == len(self.targets) else self.COLOR_TARGET_ON
            
            text_surf = self.font_end.render(msg, True, color)
            self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT // 2 - text_surf.get_height() // 2))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        movement, space, shift = 0, 0, 0
        action_to_take = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print("\n--- Game Reset ---")
                
                # Capture single key presses for actions
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_SPACE, pygame.K_LSHIFT, pygame.K_RSHIFT]:
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    if event.key == pygame.K_SPACE: space = 1
                    if event.key in [pygame.K_LSHIFT, pygame.K_RSHIFT]: shift = 1
                    action_to_take = [movement, space, shift]

        if action_to_take:
            obs, reward, terminated, truncated, info = env.step(action_to_take)
            total_reward += reward
            
            print(f"Action: {action_to_take}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Moves Left: {info['moves_left']}")

            if terminated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
                print("Press 'R' to reset.")
        
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)
        
    env.close()