
# Generated: 2025-08-28T03:45:03.855557
# Source Brief: brief_05024.md
# Brief Index: 5024

        
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
        "Controls: Use arrow keys to move the cursor. Press SHIFT to cycle through crystal types (Red, Green, Blue). "
        "Press SPACE to place the selected crystal at the cursor. Illuminate all 7 targets before time runs out!"
    )

    game_description = (
        "A minimalist puzzle game. Redirect a light beam by placing refractive crystals in an isometric cavern. "
        "Your goal is to illuminate all target gems within the time limit. Each crystal type bends light differently."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # --- Game Constants ---
        self.MAX_STEPS = self.FPS * 60  # 60 seconds
        self.GRID_SIZE = (22, 14)
        self.TILE_WIDTH = 30
        self.TILE_HEIGHT = 15
        self.ORIGIN_X = self.WIDTH // 2 - 10
        self.ORIGIN_Y = 70
        
        # --- Colors ---
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_BEAM = (255, 255, 255)
        self.COLOR_BEAM_GLOW = (180, 220, 255, 100)
        self.COLOR_TARGET_DIM = (100, 100, 120)
        self.COLOR_TARGET_LIT = (255, 255, 100)
        self.COLOR_TARGET_LIT_GLOW = (255, 255, 0, 100)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 255)

        self.CRYSTAL_DEFINITIONS = [
            {'name': 'red', 'angle': 45, 'color': (255, 80, 80), 'top_color': (255, 150, 150), 'side_color': (180, 50, 50)},
            {'name': 'green', 'angle': 90, 'color': (80, 255, 80), 'top_color': (150, 255, 150), 'side_color': (50, 180, 50)},
            {'name': 'blue', 'angle': 135, 'color': (80, 80, 255), 'top_color': (150, 150, 255), 'side_color': (50, 50, 180)},
        ]
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.cursor_pos = [0, 0]
        self.placed_crystals = []
        self.target_crystals = []
        self.beam_source = {}
        self.light_path = []
        self.particles = []
        self.selected_crystal_type_idx = 0
        self.crystal_inventory = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.win_condition = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.time_left = self.MAX_STEPS
        
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.placed_crystals = []
        self.crystal_inventory = 7
        self.selected_crystal_type_idx = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles = []
        
        self._generate_level()
        self._recalculate_all_lighting()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.beam_source = {'pos': (0, self.GRID_SIZE[1] // 2), 'dir': np.array([1.0, 0.0])}
        
        # Use a fixed, solvable layout for consistency
        target_positions = [
            (3, 2), (5, 9), (9, 4), (12, 10), (15, 1), (18, 7), (19, 11)
        ]
        self.target_crystals = [{'pos': pos, 'lit': False, 'radius': 8} for pos in target_positions]

    def step(self, action):
        reward = 0
        if not self.game_over:
            prev_lit_count = sum(1 for c in self.target_crystals if c['lit'])
            
            placement_occurred = self._handle_input(action)
            self._update_game_state()
            
            if placement_occurred:
                self._recalculate_all_lighting()
                # sfx: CRYSTAL_PLACE
            
            current_lit_count = sum(1 for c in self.target_crystals if c['lit'])
            newly_lit_count = current_lit_count - prev_lit_count
            
            if newly_lit_count > 0:
                reward += newly_lit_count * 10.0
            
            self.win_condition = current_lit_count == len(self.target_crystals)
            terminated = self.time_left <= 0 or self.win_condition
            
            if terminated:
                if self.win_condition:
                    reward += 100.0 # Win bonus
                    # sfx: WIN_SOUND
                else:
                    reward -= 100.0 # Time out penalty
                    # sfx: LOSE_SOUND
                self.game_over = True
        else:
            terminated = True

        self.score += reward
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        move_speed = 1
        if movement == 1: self.cursor_pos[1] -= move_speed
        elif movement == 2: self.cursor_pos[1] += move_speed
        elif movement == 3: self.cursor_pos[0] -= move_speed
        elif movement == 4: self.cursor_pos[0] += move_speed
        
        self.cursor_pos[0] = max(0, min(self.GRID_SIZE[0] - 1, self.cursor_pos[0]))
        self.cursor_pos[1] = max(0, min(self.GRID_SIZE[1] - 1, self.cursor_pos[1]))

        if shift_held and not self.prev_shift_held:
            self.selected_crystal_type_idx = (self.selected_crystal_type_idx + 1) % len(self.CRYSTAL_DEFINITIONS)
            # sfx: UI_CYCLE
        
        placement_occurred = False
        if space_held and not self.prev_space_held:
            if self.crystal_inventory > 0:
                is_occupied = any(c['pos'] == tuple(self.cursor_pos) for c in self.placed_crystals)
                if not is_occupied:
                    crystal_def = self.CRYSTAL_DEFINITIONS[self.selected_crystal_type_idx]
                    self.placed_crystals.append({
                        'pos': tuple(self.cursor_pos),
                        'type': crystal_def['name'],
                        'angle_rad': math.radians(crystal_def['angle']),
                    })
                    self.crystal_inventory -= 1
                    placement_occurred = True

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return placement_occurred

    def _update_game_state(self):
        self.time_left = max(0, self.time_left - 1)
        
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        
        # Spawn new particles along the beam
        if self.steps % 2 == 0 and len(self.light_path) > 1:
            for i in range(len(self.light_path) - 1):
                p1 = np.array(self.light_path[i])
                p2 = np.array(self.light_path[i+1])
                if np.linalg.norm(p2 - p1) > 1:
                    pos = p1 + (p2 - p1) * self.np_random.random()
                    vel = (p2 - p1) / np.linalg.norm(p2 - p1) * 0.5
                    self.particles.append({'pos': pos, 'vel': vel, 'life': 20, 'radius': self.np_random.random() * 1.5})

    def _recalculate_all_lighting(self):
        self._calculate_light_path()
        
        for target in self.target_crystals:
            target['lit'] = False
            target_screen_pos = self._iso_to_screen(*target['pos'])
            for i in range(len(self.light_path) - 1):
                p1, p2 = self.light_path[i], self.light_path[i+1]
                if self._line_point_collision(p1, p2, target_screen_pos, target['radius']):
                    target['lit'] = True
                    # sfx: TARGET_LIT
                    break
    
    def _calculate_light_path(self):
        path = []
        max_bounces = 25
        
        start_pos_grid = self.beam_source['pos']
        start_pos_screen = np.array(self._iso_to_screen(start_pos_grid[0], start_pos_grid[1]), dtype=float)
        
        current_pos = start_pos_screen
        current_dir = self.beam_source['dir']
        
        path.append(tuple(map(int, current_pos)))

        for _ in range(max_bounces):
            intersections = []
            
            # Wall intersections
            if current_dir[0] > 1e-6: intersections.append({'t': (self.WIDTH - current_pos[0]) / current_dir[0], 'type': 'wall', 'norm': np.array([-1.0, 0.0])})
            if current_dir[0] < -1e-6: intersections.append({'t': (0 - current_pos[0]) / current_dir[0], 'type': 'wall', 'norm': np.array([1.0, 0.0])})
            if current_dir[1] > 1e-6: intersections.append({'t': (self.HEIGHT - current_pos[1]) / current_dir[1], 'type': 'wall', 'norm': np.array([0.0, -1.0])})
            if current_dir[1] < -1e-6: intersections.append({'t': (0 - current_pos[1]) / current_dir[1], 'type': 'wall', 'norm': np.array([0.0, 1.0])})
            
            # Crystal intersections
            for crystal in self.placed_crystals:
                crystal_screen_pos = np.array(self._iso_to_screen(*crystal['pos']))
                oc = current_pos - crystal_screen_pos
                a = np.dot(current_dir, current_dir)
                b = 2 * np.dot(oc, current_dir)
                c = np.dot(oc, oc) - (self.TILE_HEIGHT * 1.2)**2 # Collision radius
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    t = (-b - math.sqrt(discriminant)) / (2*a)
                    intersections.append({'t': t, 'type': 'crystal', 'obj': crystal})
            
            valid_intersections = [i for i in intersections if i['t'] > 1e-4]
            if not valid_intersections: break
                
            closest = min(valid_intersections, key=lambda x: x['t'])
            
            intersection_point = current_pos + closest['t'] * current_dir
            path.append(tuple(map(int, intersection_point)))
            
            if closest['type'] == 'wall':
                current_dir = current_dir - 2 * np.dot(current_dir, closest['norm']) * closest['norm']
                # sfx: BEAM_BOUNCE
            elif closest['type'] == 'crystal':
                angle = closest['obj']['angle_rad']
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                dx, dy = current_dir[0], current_dir[1]
                current_dir = np.array([dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a])
                # sfx: BEAM_REFRACT
            
            current_pos = intersection_point
            
        self.light_path = path

    def _line_point_collision(self, p1, p2, point, radius):
        p1, p2, point = np.array(p1), np.array(p2), np.array(point)
        l2 = np.sum((p1 - p2)**2)
        if l2 == 0.0: return np.sum((point - p1)**2) < radius**2
        t = max(0, min(1, np.dot(point - p1, p2 - p1) / l2))
        projection = p1 + t * (p2 - p1)
        return np.sum((point - projection)**2) < radius**2

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid
        for r in range(self.GRID_SIZE[1]):
            for c in range(self.GRID_SIZE[0]):
                points = [self._iso_to_screen(c, r), self._iso_to_screen(c + 1, r), self._iso_to_screen(c + 1, r + 1), self._iso_to_screen(c, r + 1)]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw target crystals
        for target in self.target_crystals:
            pos = self._iso_to_screen(*target['pos'])
            if target['lit']:
                glow_surface = self.screen.copy()
                glow_surface.set_colorkey((0,0,0))
                pygame.gfxdraw.filled_circle(glow_surface, pos[0], pos[1], target['radius'] + 4, self.COLOR_TARGET_LIT_GLOW)
                glow_surface.set_alpha(100)
                self.screen.blit(glow_surface, (0,0))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], target['radius'], self.COLOR_TARGET_LIT)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], target['radius'], self.COLOR_TARGET_LIT)
            else:
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], target['radius'], self.COLOR_TARGET_DIM)
        
        # Draw placed crystals
        for crystal in self.placed_crystals:
            c_def = next(c for c in self.CRYSTAL_DEFINITIONS if c['name'] == crystal['type'])
            self._draw_iso_block(self.screen, crystal['pos'][0], crystal['pos'][1], c_def['color'], c_def['top_color'], c_def['side_color'])
        
        # Draw beam source
        source_pos = self._iso_to_screen(*self.beam_source['pos'])
        pygame.gfxdraw.filled_circle(self.screen, source_pos[0], source_pos[1], 5, self.COLOR_BEAM)
        pygame.gfxdraw.aacircle(self.screen, source_pos[0], source_pos[1], 5, self.COLOR_BEAM)

        # Draw light beam path
        if len(self.light_path) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_BEAM_GLOW, False, self.light_path, 5)
            pygame.draw.aalines(self.screen, self.COLOR_BEAM, False, self.light_path, 2)
        
        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20))
            color = (*self.COLOR_BEAM, alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

        # Draw cursor
        cursor_def = self.CRYSTAL_DEFINITIONS[self.selected_crystal_type_idx]
        c_pos = self.cursor_pos
        points = [self._iso_to_screen(c_pos[0], c_pos[1]), self._iso_to_screen(c_pos[0] + 1, c_pos[1]), self._iso_to_screen(c_pos[0] + 1, c_pos[1] + 1), self._iso_to_screen(c_pos[0], c_pos[1] + 1)]
        pygame.draw.lines(self.screen, (*cursor_def['color'], 150), True, points, 3)

    def _render_ui(self):
        # Timer
        time_str = f"TIME: {self.time_left / self.FPS:.1f}"
        time_surf = self.font_main.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Score
        score_str = f"SCORE: {int(self.score)}"
        score_surf = self.font_main.render(score_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (10, 10))

        # Crystal Inventory & Selection
        inv_str = f"CRYSTALS: {self.crystal_inventory}/7"
        inv_surf = self.font_main.render(inv_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(inv_surf, (10, self.HEIGHT - 30))
        
        selected_def = self.CRYSTAL_DEFINITIONS[self.selected_crystal_type_idx]
        sel_str = f"SELECTED: {selected_def['name'].upper()}"
        sel_surf = self.font_main.render(sel_str, True, selected_def['color'])
        self.screen.blit(sel_surf, (10, self.HEIGHT - 55))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "SUCCESS" if self.win_condition else "TIME UP"
            color = self.COLOR_TARGET_LIT if self.win_condition else self.COLOR_UI_TEXT
            msg_surf = self.font_large.render(msg, True, color)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def _iso_to_screen(self, x, y):
        screen_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH / 2
        screen_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT / 2
        return int(screen_x), int(screen_y)
        
    def _draw_iso_block(self, surface, x, y, color, top_color, side_color):
        h = 0.5 
        top_points = [self._iso_to_screen(x, y), self._iso_to_screen(x + 1, y), self._iso_to_screen(x + 1, y + 1), self._iso_to_screen(x, y + 1)]
        side_1 = [self._iso_to_screen(x, y + 1), self._iso_to_screen(x + 1, y + 1), self._iso_to_screen(x + 1, y + 1 + h), self._iso_to_screen(x, y + 1 + h)]
        side_2 = [self._iso_to_screen(x + 1, y), self._iso_to_screen(x + 1, y + 1), self._iso_to_screen(x + 1, y + 1 + h), self._iso_to_screen(x + 1, y + h)]

        pygame.gfxdraw.filled_polygon(surface, side_1, side_color)
        pygame.gfxdraw.filled_polygon(surface, side_2, side_color)
        pygame.gfxdraw.filled_polygon(surface, top_points, top_color)
        
        pygame.gfxdraw.aapolygon(surface, top_points, color)
        pygame.gfxdraw.aapolygon(surface, side_1, color)
        pygame.gfxdraw.aapolygon(surface, side_2, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "crystals_placed": len(self.placed_crystals),
            "targets_lit": sum(1 for c in self.target_crystals if c['lit'])
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Reset action
        action.fill(0)
        action[0] = 0 # No movement
        action[1] = 0 # Space released
        action[2] = 0 # Shift released

        # Map keys to actions
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}")
            print("Resetting in 3 seconds...")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
        
        env.clock.tick(env.FPS)
        
    env.close()