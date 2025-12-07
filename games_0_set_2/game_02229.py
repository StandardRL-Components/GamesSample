
# Generated: 2025-08-28T04:14:26.736570
# Source Brief: brief_02229.md
# Brief Index: 2229

        
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
        "Controls: Use arrow keys to move the selected crystal. Press Space to cycle through crystals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a shimmering crystal maze, strategically shifting crystals to illuminate pathways and escape the caverns before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 14, 10
        self.MAX_STEPS = 600
        self.NUM_PATHS = 20
        self.NUM_CRYSTALS = 6 # 2 of each color

        # Colors
        self.COLOR_BG = (20, 25, 35)
        self.CRYSTAL_COLORS = {
            'R': (255, 50, 50),
            'G': (50, 255, 50),
            'B': (50, 100, 255)
        }
        self.PATH_COLORS = {
            0: (80, 80, 100),    # Unlit
            1: (255, 255, 0),    # Partially lit
            2: (200, 255, 255)   # Fully lit
        }
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_SELECTION = (255, 255, 150)
        self.TARGET_MARKER_COLOR = (60, 60, 75)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 50)

        # Isometric projection helpers
        self.tile_w = 32
        self.tile_h = 16
        self.origin_x = self.WIDTH // 2
        self.origin_y = 80
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.remaining_steps = 0
        self.crystals = []
        self.path_definitions = []
        self.paths = []
        self.selected_crystal_idx = 0
        self.prev_space_held = False
        self.np_random = None

        # This method must be called before reset to populate path rules
        self._define_paths()
        
        # Initialize state variables
        self.reset()
        
        # Run validation check at the end of init
        self.validate_implementation()

    def _define_paths(self):
        """Pre-generates a fixed set of path rules for consistency across episodes."""
        # Use a fixed seed for path generation to ensure the puzzle is the same every time
        path_rng = random.Random(12345)
        crystal_ids = list(range(self.NUM_CRYSTALS))
        
        self.path_definitions = []
        used_rules = set()

        # 10 single-crystal paths
        for _ in range(10):
            while True:
                crystal_id = path_rng.choice(crystal_ids)
                pos = (path_rng.randint(0, self.GRID_WIDTH - 1), path_rng.randint(0, self.GRID_HEIGHT - 1))
                rule_tuple = (crystal_id, pos)
                if rule_tuple not in used_rules:
                    used_rules.add(rule_tuple)
                    self.path_definitions.append({
                        'rules': [{'crystal_id': crystal_id, 'pos': pos}],
                        'endpoints': [pos, pos]
                    })
                    break
        
        # 10 dual-crystal paths
        for _ in range(10):
            while True:
                c1_id, c2_id = path_rng.sample(crystal_ids, 2)
                pos1 = (path_rng.randint(0, self.GRID_WIDTH - 1), path_rng.randint(0, self.GRID_HEIGHT - 1))
                pos2 = (path_rng.randint(0, self.GRID_WIDTH - 1), path_rng.randint(0, self.GRID_HEIGHT - 1))
                if pos1 == pos2: continue
                
                rule1_tuple = (c1_id, pos1)
                rule2_tuple = (c2_id, pos2)

                # Ensure rules don't conflict with existing ones
                if rule1_tuple not in used_rules and rule2_tuple not in used_rules and \
                   (c2_id, pos1) not in used_rules and (c1_id, pos2) not in used_rules:
                    used_rules.add(rule1_tuple)
                    used_rules.add(rule2_tuple)
                    self.path_definitions.append({
                        'rules': [
                            {'crystal_id': c1_id, 'pos': pos1},
                            {'crystal_id': c2_id, 'pos': pos2}
                        ],
                        'endpoints': [pos1, pos2]
                    })
                    break
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.remaining_steps = self.MAX_STEPS
        self.selected_crystal_idx = 0
        self.prev_space_held = False

        # Create crystals
        colors = ['R', 'R', 'G', 'G', 'B', 'B']
        self.crystals = [{'id': i, 'color': colors[i], 'pos': (0, 0)} for i in range(self.NUM_CRYSTALS)]

        # Place crystals randomly without overlap
        occupied_positions = set()
        for crystal in self.crystals:
            while True:
                pos = (self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT))
                if pos not in occupied_positions:
                    crystal['pos'] = pos
                    occupied_positions.add(pos)
                    break
        
        self.paths = [{'state': 0, 'rules': d['rules'], 'endpoints': d['endpoints']} for d in self.path_definitions]
        self._update_path_lighting()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        old_fully_lit_count = sum(1 for p in self.paths if p['state'] == 2)

        # Handle actions
        if space_held and not self.prev_space_held:
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % len(self.crystals)
            # Sound effect placeholder: select_sound.play()

        if movement > 0:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][movement - 1]
            selected_crystal = self.crystals[self.selected_crystal_idx]
            current_pos = selected_crystal['pos']
            target_pos = (current_pos[0] + dx, current_pos[1] + dy)

            if 0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT:
                if not any(c['pos'] == target_pos for c in self.crystals):
                    selected_crystal['pos'] = target_pos
                    # Sound effect placeholder: move_sound.play()

        self.prev_space_held = space_held
        
        # Update game logic
        self._update_path_lighting()
        self.steps += 1
        self.remaining_steps -= 1

        reward = self._calculate_reward(old_fully_lit_count)
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if sum(1 for p in self.paths if p['state'] == 2) == self.NUM_PATHS:
                reward += 100  # Win bonus
                # Sound effect placeholder: win_sound.play()
            else:
                reward -= 100  # Loss penalty
                # Sound effect placeholder: lose_sound.play()
        
        self.score += reward

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _update_path_lighting(self):
        for path in self.paths:
            rules_satisfied = sum(1 for rule in path['rules'] if self.crystals[rule['crystal_id']]['pos'] == rule['pos'])
            
            num_rules = len(path['rules'])
            new_state = 0
            if rules_satisfied == num_rules and num_rules > 0:
                new_state = 2  # Fully lit
            elif rules_satisfied > 0:
                new_state = 1  # Partially lit
            
            if new_state == 2 and path['state'] != 2:
                # Sound effect placeholder: path_lit_sound.play()
                pass
            path['state'] = new_state

    def _calculate_reward(self, old_fully_lit_count):
        reward = 0.0
        new_fully_lit_count = 0
        for path in self.paths:
            if path['state'] == 1: reward += 0.1
            elif path['state'] == 2:
                reward += 1.0
                new_fully_lit_count += 1
        
        newly_lit = new_fully_lit_count - old_fully_lit_count
        if newly_lit > 0: reward += newly_lit * 5.0
            
        return reward

    def _check_termination(self):
        all_lit = all(p['state'] == 2 for p in self.paths)
        time_up = self.remaining_steps <= 0
        return all_lit or time_up

    def _grid_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * (self.tile_w / 2)
        iso_y = self.origin_y + (x + y) * (self.tile_h / 2)
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, x, y, color, size_mod=0):
        iso_x, iso_y = self._grid_to_iso(x, y)
        w, h = self.tile_w + size_mod, self.tile_h + size_mod
        
        points = [
            (iso_x, iso_y),
            (iso_x + w / 2, iso_y + h / 2),
            (iso_x, iso_y + h),
            (iso_x - w / 2, iso_y + h / 2),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, color)
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render path target locations and lines
        for path in self.paths:
            for rule in path['rules']:
                tx, ty = rule['pos']
                self._draw_iso_cube(self.screen, tx, ty, self.TARGET_MARKER_COLOR)
            
            start_pos = self._grid_to_iso(*path['endpoints'][0])
            end_pos = self._grid_to_iso(*path['endpoints'][1])
            start_pos = (start_pos[0], start_pos[1] + self.tile_h / 2)
            end_pos = (end_pos[0], end_pos[1] + self.tile_h / 2)
            
            path_color = self.PATH_COLORS[path['state']]
            path_width = 1 if path['state'] < 2 else 3
            pygame.draw.aaline(self.screen, path_color, start_pos, end_pos)
            if path_width > 1:
                 pygame.draw.line(self.screen, path_color, start_pos, end_pos, path_width)

        # Render crystals
        for i, crystal in enumerate(self.crystals):
            x, y = crystal['pos']
            color = self.CRYSTAL_COLORS[crystal['color']]
            
            # Selection highlight
            if i == self.selected_crystal_idx and not self.game_over:
                pulse = (1 + math.sin(self.steps * 0.4)) * 5
                self._draw_iso_cube(self.screen, x, y, self.COLOR_SELECTION, size_mod=pulse)
            
            self._draw_iso_cube(self.screen, x, y, color)

    def _render_ui(self):
        lit_count = sum(1 for p in self.paths if p['state'] == 2)
        path_text = f"Paths Lit: {lit_count} / {self.NUM_PATHS}"
        text_surface = self.font_ui.render(path_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surface, (10, 10))
        
        time_text = f"Time: {self.remaining_steps}"
        text_surface = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surface.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(text_surface, text_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            all_lit = lit_count == self.NUM_PATHS
            msg = "SUCCESS!" if all_lit else "TIME UP"
            color = (150, 255, 150) if all_lit else (255, 150, 150)
            
            text_surface = self.font_title.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_steps": self.remaining_steps,
            "lit_paths": sum(1 for p in self.paths if p['state'] == 2),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Crystal Maze")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    last_action_time = pygame.time.get_ticks()
    action_delay = 100 # ms between actions for human play

    while running:
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # For turn-based, we only care about key down events
            if event.type == pygame.KEYDOWN:
                if pygame.time.get_ticks() - last_action_time > action_delay:
                    if event.key == pygame.K_UP: movement = 1
                    elif event.key == pygame.K_DOWN: movement = 2
                    elif event.key == pygame.K_LEFT: movement = 3
                    elif event.key == pygame.K_RIGHT: movement = 4
                    elif event.key == pygame.K_SPACE: space_held = 1
                    
                    action = [movement, space_held, shift_held]
                    
                    if any(a != 0 for a in action):
                        obs, reward, terminated, truncated, info = env.step(action)
                        last_action_time = pygame.time.get_ticks()
                        if terminated:
                            print(f"Game Over! Final Score: {info['score']:.2f}")
                            # Render one last time to show final state
                            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                            screen.blit(surf, (0, 0))
                            pygame.display.flip()
                            pygame.time.wait(3000)
                            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
        
    env.close()