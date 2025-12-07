import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Space to place a crystal. Shift to cycle crystal type."
    )

    game_description = (
        "Navigate a crystal cavern, placing crystals to redirect light beams and illuminate all targets before time runs out."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = 20
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    NUM_TARGETS = 10
    MAX_CRYSTALS = 20
    MAX_BEAM_BOUNCES = 15

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_WALL = (40, 50, 70)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_TARGET = (120, 130, 150)
    COLOR_TARGET_LIT = (255, 255, 255)
    COLOR_TARGET_GLOW = (200, 255, 255, 50)
    COLOR_SOURCE = (255, 255, 0)
    COLOR_BEAM = (255, 255, 0)
    CRYSTAL_COLORS = [(255, 80, 80), (80, 255, 80), (80, 80, 255)] # Red, Green, Blue
    CRYSTAL_ANGLES = [math.radians(45), math.radians(90), math.radians(135)] # Refraction angles
    UI_TEXT_COLOR = (220, 220, 240)
    UI_BG_COLOR = (30, 40, 60, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        self.game_state_initialized = False
        # self.reset() is called by the validation method, which is good practice.
        # self.validate_implementation() # This will be called outside after init

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, 'np_random') or self.np_random is None:
             self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.time_remaining = self.MAX_STEPS

        self.cursor_pos = np.array([self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2], dtype=float)
        self.selected_crystal_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_level()

        self.crystals = []
        self.particles = []
        self.light_beams = []
        self.num_lit_targets_last_step = 0

        self._update_light_beams()
        self.game_state_initialized = True
        return self._get_observation(), self._get_info()

    def step(self, action):
        if not self.game_state_initialized:
            # This should not happen if reset() is called in __init__
            obs, info = self.reset()
            return obs, 0, False, False, info
            
        reward = -0.01  # Small penalty for time passing
        self.time_remaining -= 1
        self.steps += 1

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        beams_need_update = self._handle_input(movement, space_held, shift_held)

        if beams_need_update:
            # Reward for placing a crystal is calculated after beams are updated
            placement_reward = self._update_light_beams()
            reward += placement_reward
            # SFX: place_crystal.wav

        self._update_particles()

        # Reward for lighting new targets
        current_lit_targets = sum(1 for t in self.targets if t['lit'])
        if current_lit_targets > self.num_lit_targets_last_step:
            newly_lit = current_lit_targets - self.num_lit_targets_last_step
            reward += newly_lit * 10.0
            self.score += newly_lit * 10
            for target in self.targets:
                if target['lit'] and not target['was_lit']:
                    self._create_particles(target['pos'], self.COLOR_TARGET_LIT, 20)
                    # SFX: target_lit.wav
            
        self.num_lit_targets_last_step = current_lit_targets
        for target in self.targets:
            target['was_lit'] = target['lit']

        terminated = self._check_termination()
        if terminated:
            if self.win_condition:
                reward += 100
                self.score += 100
            else: # Time ran out
                reward -= 100
                self.score -= 100

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        cursor_speed = 5
        if movement == 1: self.cursor_pos[1] -= cursor_speed
        elif movement == 2: self.cursor_pos[1] += cursor_speed
        elif movement == 3: self.cursor_pos[0] -= cursor_speed
        elif movement == 4: self.cursor_pos[0] += cursor_speed
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT - 1)

        beams_need_update = False
        
        # --- Cycle Crystal (on key press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_COLORS)
            self._create_particles(self.cursor_pos, self.CRYSTAL_COLORS[self.selected_crystal_type], 10, speed=2)
            # SFX: cycle_type.wav

        # --- Place Crystal (on key press) ---
        if space_held and not self.prev_space_held and len(self.crystals) < self.MAX_CRYSTALS:
            grid_x, grid_y = int(self.cursor_pos[0] // self.CELL_SIZE), int(self.cursor_pos[1] // self.CELL_SIZE)
            if self.grid[grid_y, grid_x] == 0: # Can only place in empty space
                self.grid[grid_y, grid_x] = 2 + self.selected_crystal_type # Mark grid
                self.crystals.append({
                    'pos': (grid_x * self.CELL_SIZE + self.CELL_SIZE // 2, grid_y * self.CELL_SIZE + self.CELL_SIZE // 2),
                    'type': self.selected_crystal_type,
                    'angle': self.CRYSTAL_ANGLES[self.selected_crystal_type],
                    'is_hit': False
                })
                beams_need_update = True

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return beams_need_update

    def _generate_level(self):
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        
        # Place walls around the border
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        # Add some random inner walls
        num_walls = self.np_random.integers(15, 30)
        for _ in range(num_walls):
            y, x = self.np_random.integers(1, [self.GRID_ROWS-1, self.GRID_COLS-1])
            self.grid[y, x] = 1
        
        # Get all valid empty cells
        empty_cells = list(zip(*np.where(self.grid == 0)))
        self.np_random.shuffle(empty_cells)
        
        # Place light source on an edge
        edge_cells = []
        for r in range(1, self.GRID_ROWS - 1):
            if self.grid[r, 1] == 0: edge_cells.append((r, 1))
            if self.grid[r, self.GRID_COLS - 2] == 0: edge_cells.append((r, self.GRID_COLS - 2))
        for c in range(1, self.GRID_COLS - 1):
            if self.grid[1, c] == 0: edge_cells.append((1, c))
            if self.grid[self.GRID_ROWS - 2, c] == 0: edge_cells.append((self.GRID_ROWS - 2, c))
        
        self.np_random.shuffle(edge_cells)
        source_y, source_x = edge_cells.pop(0)
        
        self.grid[source_y, source_x] = -1 # Mark source
        self.light_source = {
            'pos': np.array([source_x * self.CELL_SIZE + self.CELL_SIZE/2, source_y * self.CELL_SIZE + self.CELL_SIZE/2]),
            'dir': self._get_initial_beam_dir((source_y, source_x))
        }

        # Place targets
        self.targets = []
        possible_target_cells = [cell for cell in empty_cells if self.grid[cell] == 0 and cell != (source_y, source_x)]
        self.np_random.shuffle(possible_target_cells)
        num_to_place = min(self.NUM_TARGETS, len(possible_target_cells))
        
        for i in range(num_to_place):
            y, x = possible_target_cells[i]
            self.grid[y, x] = -2 # Mark target
            self.targets.append({
                'pos': (x * self.CELL_SIZE + self.CELL_SIZE // 2, y * self.CELL_SIZE + self.CELL_SIZE // 2),
                'lit': False,
                'was_lit': False
            })

    def _get_initial_beam_dir(self, pos):
        y, x = pos
        if y == 1: return np.array([0, 1])
        if y == self.GRID_ROWS - 2: return np.array([0, -1])
        if x == 1: return np.array([1, 0])
        if x == self.GRID_COLS - 2: return np.array([-1, 0])
        # Default case if somehow not on an edge
        return np.array([1, 0])

    def _update_light_beams(self):
        self.light_beams = []
        for crystal in self.crystals: crystal['is_hit'] = False
        
        q = [(self.light_source['pos'], self.light_source['dir'], 0)]
        
        visited_crystal_interactions = set()

        while q:
            pos, direction, bounce_count = q.pop(0)
            
            if bounce_count > self.MAX_BEAM_BOUNCES: continue

            start_pos = np.copy(pos)
            
            # Ray marching
            current_pos = np.copy(start_pos)
            for _ in range(int(self.SCREEN_WIDTH * 1.5)): # Max trace distance
                current_pos += direction
                
                grid_x, grid_y = int(current_pos[0] // self.CELL_SIZE), int(current_pos[1] // self.CELL_SIZE)

                if not (0 <= grid_y < self.GRID_ROWS and 0 <= grid_x < self.GRID_COLS):
                    break # Hit screen boundary

                cell_content = self.grid[grid_y, grid_x]
                
                if cell_content == 1: # Wall
                    break
                
                if cell_content >= 2: # Crystal
                    crystal_index = -1
                    crystal_center = (grid_x * self.CELL_SIZE + self.CELL_SIZE // 2, grid_y * self.CELL_SIZE + self.CELL_SIZE // 2)
                    for i, c in enumerate(self.crystals):
                        if c['pos'] == crystal_center:
                            crystal_index = i
                            break
                    
                    if crystal_index != -1:
                        crystal = self.crystals[crystal_index]
                        crystal['is_hit'] = True
                        
                        interaction_key = (crystal_index, tuple(np.round(direction, 3)))
                        if interaction_key in visited_crystal_interactions:
                            break # Infinite loop detected
                        visited_crystal_interactions.add(interaction_key)

                        # Rotate direction vector
                        angle = crystal['angle']
                        new_dx = direction[0] * math.cos(angle) - direction[1] * math.sin(angle)
                        new_dy = direction[0] * math.sin(angle) + direction[1] * math.cos(angle)
                        new_dir = np.array([new_dx, new_dy])
                        
                        q.append((current_pos, new_dir, bounce_count + 1))
                        break

            self.light_beams.append((start_pos, current_pos))

        # Check for lit targets
        for target in self.targets:
            target['lit'] = False
            for start, end in self.light_beams:
                dist = self._point_segment_distance(np.array(target['pos']), start, end)
                if dist < self.CELL_SIZE / 2:
                    target['lit'] = True
                    break
        
        # Calculate reward for crystal placement
        placement_reward = 0
        if self.crystals:
            last_crystal = self.crystals[-1]
            if not last_crystal['is_hit']:
                placement_reward -= 1.0 # Punish useless crystal
            else:
                placement_reward += 0.5 # Reward useful crystal
        return placement_reward

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        all_targets_lit = len(self.targets) > 0 and all(t['lit'] for t in self.targets)
        if all_targets_lit:
            self.win_condition = True
            self.game_over = True
            return True
            
        if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
            
        return False

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
            "time_remaining": self.time_remaining / self.FPS,
            "lit_targets": sum(1 for t in self.targets if t['lit']),
            "total_targets": len(self.targets)
        }
        
    def _render_game(self):
        # Walls
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Targets
        for target in self.targets:
            pos_int = (int(target['pos'][0]), int(target['pos'][1]))
            if target['lit']:
                # Glow effect
                glow_radius = int(self.CELL_SIZE * 0.8 * (0.8 + 0.2 * math.sin(self.steps * 0.2)))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_radius, self.COLOR_TARGET_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.CELL_SIZE // 3, self.COLOR_TARGET_LIT)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.CELL_SIZE // 3, self.COLOR_TARGET_LIT)
            else:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.CELL_SIZE // 3, self.COLOR_TARGET)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.CELL_SIZE // 3, self.COLOR_TARGET)
        
        # Light Source
        src_pos = (int(self.light_source['pos'][0]), int(self.light_source['pos'][1]))
        pulse = (math.sin(self.steps * 0.3) + 1) / 2
        radius = int(self.CELL_SIZE * 0.4 + pulse * 3)
        pygame.gfxdraw.filled_circle(self.screen, src_pos[0], src_pos[1], radius, self.COLOR_SOURCE)
        pygame.gfxdraw.aacircle(self.screen, src_pos[0], src_pos[1], radius, self.COLOR_SOURCE)

        # Light Beams
        beam_alpha = 150 + 105 * math.sin(self.steps * 0.4)
        beam_color = (*self.COLOR_BEAM, beam_alpha)
        for start, end in self.light_beams:
            pygame.draw.line(self.screen, beam_color, start, end, 3)

        # Crystals
        for crystal in self.crystals:
            pos_int = (int(crystal['pos'][0]), int(crystal['pos'][1]))
            color = self.CRYSTAL_COLORS[crystal['type']]
            size = self.CELL_SIZE // 3
            points = [
                (pos_int[0], pos_int[1] - size),
                (pos_int[0] + size, pos_int[1]),
                (pos_int[0], pos_int[1] + size),
                (pos_int[0] - size, pos_int[1]),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

        # Particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['life'] / p['max_life'])))
            color = (*p['color'], alpha)
            size = int(p['size'] * (p['life'] / p['max_life']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), size, color)
        
        # Cursor
        if not self.game_over:
            cursor_int = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
            color = self.CRYSTAL_COLORS[self.selected_crystal_type]
            size = self.CELL_SIZE // 3
            points = [
                (cursor_int[0], cursor_int[1] - size),
                (cursor_int[0] + size, cursor_int[1]),
                (cursor_int[0], cursor_int[1] + size),
                (cursor_int[0] - size, cursor_int[1]),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, (*color, 128))
            pygame.gfxdraw.aapolygon(self.screen, points, (*color, 200))

    def _render_ui(self):
        # UI Background Panel
        panel_rect = pygame.Rect(self.SCREEN_WIDTH - 220, 5, 215, 60)
        s = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        s.fill(self.UI_BG_COLOR)
        self.screen.blit(s, panel_rect.topleft)
        pygame.draw.rect(self.screen, self.UI_TEXT_COLOR, panel_rect, 1, 3)

        # Timer
        time_sec = max(0, self.time_remaining / self.FPS)
        time_text = f"TIME: {int(time_sec // 60):02}:{int(time_sec % 60):02}"
        time_surf = self.font_main.render(time_text, True, self.UI_TEXT_COLOR)
        self.screen.blit(time_surf, (self.SCREEN_WIDTH - 210, 15))

        # Target Counter
        lit_count = sum(1 for t in self.targets if t['lit'])
        target_text = f"TARGETS: {lit_count} / {len(self.targets)}"
        target_surf = self.font_main.render(target_text, True, self.UI_TEXT_COLOR)
        self.screen.blit(target_surf, (self.SCREEN_WIDTH - 210, 40))

        # Selected Crystal UI
        ui_crystal_pos = [35, self.SCREEN_HEIGHT - 35]
        for i in range(3):
            color = self.CRYSTAL_COLORS[i]
            is_selected = (i == self.selected_crystal_type)
            size = self.CELL_SIZE // 2 if is_selected else self.CELL_SIZE // 3
            pos = (ui_crystal_pos[0] + i * 40, ui_crystal_pos[1])
            points = [ (pos[0], pos[1] - size), (pos[0] + size, pos[1]), (pos[0], pos[1] + size), (pos[0] - size, pos[1])]
            alpha_color = (*color, 255 if is_selected else 100)
            pygame.gfxdraw.filled_polygon(self.screen, points, alpha_color)
            pygame.gfxdraw.aapolygon(self.screen, points, alpha_color)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_condition else "TIME'S UP"
            color = self.COLOR_TARGET_LIT if self.win_condition else self.CRYSTAL_COLORS[0]
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _create_particles(self, pos, color, count, speed=4, life=15, size=3):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(0.5, 1) * speed
            vel = np.array([math.cos(angle), math.sin(angle)]) * vel_mag
            self.particles.append({
                'pos': np.copy(pos).astype(float),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life + 1),
                'max_life': life,
                'color': color,
                'size': size
            })

    @staticmethod
    def _point_segment_distance(p, a, b):
        if np.array_equal(a, b): return np.linalg.norm(p - a)
        l2 = np.linalg.norm(b - a)**2
        if l2 == 0: return np.linalg.norm(p - a)
        t = max(0, min(1, np.dot(p - a, b - a) / l2))
        projection = a + t * (b - a)
        return np.linalg.norm(p - projection)
    
    def validate_implementation(self):
        # This is a helper function to quickly check for API compliance.
        print("✓ Starting implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        print("✓ Reset method validated.")
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert obs.dtype == np.uint8
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool) and not trunc
        assert isinstance(info, dict)
        print("✓ Step method validated.")
        
        print("✓ Implementation validated successfully.")

if __name__ == "__main__":
    # To run and play the game manually
    # This requires a display. Set SDL_VIDEODRIVER to your display driver.
    # E.g., `unset SDL_VIDEODRIVER` or `export SDL_VIDEODRIVER=x11`
    if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # Run validation
    env.validate_implementation()

    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(GameEnv.user_guide)

    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    pygame.quit()