
# Generated: 2025-08-27T18:52:47.464874
# Source Brief: brief_01978.md
# Brief Index: 1978

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the cursor. Press Space to select/deselect a crystal. "
        "When a crystal is selected, use arrows to move it. Hold Shift to reset the level."
    )

    game_description = (
        "A turn-based isometric puzzle game. Move shimmering crystals to match the "
        "ghostly target pattern. Each move slides a crystal until it hits an obstacle. "
        "Solve all 15 levels before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.rng = np.random.default_rng()

        # --- Visuals ---
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 48, bold=True)

        self.COLOR_BG = (26, 28, 44)
        self.COLOR_GRID = (58, 60, 76)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_SUCCESS = (100, 255, 120)
        self.COLOR_FAIL = (255, 100, 100)

        self.CRYSTAL_COLORS = {
            "omni": ((255, 79, 121), (255, 150, 175), (180, 50, 80)),  # Red
            "horiz": ((79, 219, 255), (150, 235, 255), (50, 150, 180)), # Blue
            "vert": ((86, 255, 136), (160, 255, 180), (50, 180, 80)),  # Green
        }
        self.TILE_WIDTH_HALF = 28
        self.TILE_HEIGHT_HALF = 14
        self.CRYSTAL_HEIGHT = 24

        # --- Game Data ---
        self._generate_levels()
        
        # --- State Variables ---
        self.current_level_index = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.level_complete_flash = 0
        self.game_won_flash = 0

        self.reset()
        # self.validate_implementation() # Optional validation call

    def _generate_levels(self):
        self.levels = []
        # Lvl 1: Simple move
        self.levels.append({
            "size": (5, 5), "moves": 3,
            "crystals": [{"id": 0, "pos": (1, 2), "type": "omni"}],
            "targets": [{"id": 0, "pos": (3, 2)}]
        })
        # Lvl 2: Two crystals
        self.levels.append({
            "size": (5, 5), "moves": 4,
            "crystals": [{"id": 0, "pos": (1, 1), "type": "omni"}, {"id": 1, "pos": (3, 3), "type": "omni"}],
            "targets": [{"id": 0, "pos": (3, 3)}, {"id": 1, "pos": (1, 1)}]
        })
        # Lvl 3: Introduce constraints
        self.levels.append({
            "size": (6, 6), "moves": 4,
            "crystals": [{"id": 0, "pos": (1, 2), "type": "horiz"}, {"id": 1, "pos": (4, 4), "type": "vert"}],
            "targets": [{"id": 0, "pos": (4, 2)}, {"id": 1, "pos": (4, 1)}]
        })
        # Lvl 4: A simple block
        self.levels.append({
            "size": (7, 7), "moves": 5,
            "crystals": [{"id": 0, "pos": (1, 3), "type": "omni"}, {"id": 1, "pos": (3, 3), "type": "omni"}],
            "targets": [{"id": 0, "pos": (5, 3)}, {"id": 1, "pos": (3, 3)}]
        })
        # Lvl 5: Maze-like
        self.levels.append({
            "size": (7, 7), "moves": 6,
            "crystals": [
                {"id": 0, "pos": (1, 1), "type": "omni"}, {"id": 1, "pos": (3, 1), "type": "omni"},
                {"id": 2, "pos": (5, 1), "type": "omni"}, {"id": 3, "pos": (1, 5), "type": "omni"}
            ],
            "targets": [{"id": 0, "pos": (1, 3)}, {"id": 1, "pos": (3, 1)}, {"id": 2, "pos": (5, 1)}, {"id": 3, "pos": (1, 5)}]
        })
        # Add more levels procedurally
        for i in range(5, 15):
            size_w = min(8, 5 + i // 3)
            size_h = min(8, 5 + i // 3)
            num_crystals = min(5, 2 + i // 3)
            moves = 3 + i // 2
            
            crystals = []
            targets = []
            occupied = set()

            for j in range(num_crystals):
                # Gen start pos
                while True:
                    pos = (self.rng.integers(0, size_w), self.rng.integers(0, size_h))
                    if pos not in occupied:
                        occupied.add(pos)
                        break
                # Gen target pos
                while True:
                    t_pos = (self.rng.integers(0, size_w), self.rng.integers(0, size_h))
                    if t_pos not in occupied:
                        occupied.add(t_pos)
                        break
                
                crystal_type = self.rng.choice(["omni", "horiz", "vert"])
                crystals.append({"id": j, "pos": pos, "type": crystal_type})
                targets.append({"id": j, "pos": t_pos})

            self.levels.append({"size": (size_w, size_h), "moves": moves, "crystals": crystals, "targets": targets})

    def _load_level(self, level_index):
        if level_index >= len(self.levels):
            self.game_over = True
            self.game_won_flash = 30 # Frames to flash
            return

        level_data = self.levels[level_index]
        self.current_level_index = level_index
        self.grid_size = level_data["size"]
        self.moves_remaining = level_data["moves"]
        self.max_moves = level_data["moves"]

        self.crystals = [c.copy() for c in level_data["crystals"]]
        self.targets = [t.copy() for t in level_data["targets"]]
        # Sort for consistent distance calculation
        self.crystals.sort(key=lambda c: c["id"])
        self.targets.sort(key=lambda t: t["id"])

        self.cursor_pos = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        self.selected_crystal_idx = None
        self.game_phase = 'SELECT'
        self.level_complete_flash = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
            self._generate_levels() # Re-generate levels with new seed

        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_won_flash = 0
        
        self._load_level(0)

        return self._get_observation(), self._get_info()
    
    def _reset_level(self):
        # Resets current level state, with penalty
        self.score -= 10
        self._load_level(self.current_level_index)

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01 # Small cost for taking a step
        terminated = False

        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        if shift_pressed:
            self._reset_level()
            reward -= 5 # Penalty for resetting
            return self._get_observation(), reward, False, False, self._get_info()

        if self.game_phase == 'SELECT':
            if movement == 1: self.cursor_pos = (self.cursor_pos[0], max(0, self.cursor_pos[1] - 1))
            elif movement == 2: self.cursor_pos = (self.cursor_pos[0], min(self.grid_size[1] - 1, self.cursor_pos[1] + 1))
            elif movement == 3: self.cursor_pos = (max(0, self.cursor_pos[0] - 1), self.cursor_pos[1])
            elif movement == 4: self.cursor_pos = (min(self.grid_size[0] - 1, self.cursor_pos[0] + 1), self.cursor_pos[1])
            
            if space_pressed:
                for i, crystal in enumerate(self.crystals):
                    if crystal["pos"] == self.cursor_pos:
                        self.selected_crystal_idx = i
                        self.game_phase = 'DIRECT'
                        # sfx: select_crystal.wav
                        break
        
        elif self.game_phase == 'DIRECT':
            if space_pressed: # Cancel selection
                self.selected_crystal_idx = None
                self.game_phase = 'SELECT'
                # sfx: deselect.wav
            
            elif movement != 0: # This is a move command
                move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
                crystal = self.crystals[self.selected_crystal_idx]

                # Check if move is allowed by crystal type
                is_valid_move_type = (
                    crystal["type"] == "omni" or
                    (crystal["type"] == "horiz" and move_dir[1] == 0) or
                    (crystal["type"] == "vert" and move_dir[0] == 0)
                )

                if is_valid_move_type:
                    old_dist = self._calculate_total_distance()
                    
                    moved = self._execute_slide(self.selected_crystal_idx, move_dir)
                    
                    if moved:
                        self.moves_remaining -= 1
                        # sfx: crystal_slide.wav
                        new_dist = self._calculate_total_distance()
                        
                        # Reward for distance change
                        reward += (old_dist - new_dist)
                        self.score += (old_dist - new_dist)

                        if self._check_level_complete():
                            reward += 5
                            self.score += 25
                            self.level_complete_flash = 30 # frames
                            # sfx: level_complete.wav
                            if self.current_level_index == len(self.levels) - 1:
                                reward += 100
                                self.score += 1000
                                terminated = True
                                self.game_over = True
                                self.game_won_flash = 30
                                # sfx: game_win.wav
                            else:
                                self._load_level(self.current_level_index + 1)
                        
                        elif self.moves_remaining <= 0:
                            reward = -100
                            self.score -= 100
                            terminated = True
                            self.game_over = True
                            # sfx: game_over.wav

                # Reset phase for next turn
                self.game_phase = 'SELECT'
                self.selected_crystal_idx = None

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _execute_slide(self, crystal_idx, move_dir):
        crystal = self.crystals[crystal_idx]
        start_pos = crystal["pos"]
        current_pos = list(start_pos)
        
        other_crystal_pos = {c["pos"] for i, c in enumerate(self.crystals) if i != crystal_idx}

        while True:
            next_pos = (current_pos[0] + move_dir[0], current_pos[1] + move_dir[1])
            
            # Check boundaries
            if not (0 <= next_pos[0] < self.grid_size[0] and 0 <= next_pos[1] < self.grid_size[1]):
                break
            # Check collisions with other crystals
            if next_pos in other_crystal_pos:
                break
            
            current_pos[0], current_pos[1] = next_pos[0], next_pos[1]

        final_pos = tuple(current_pos)
        if final_pos != start_pos:
            crystal["pos"] = final_pos
            return True
        return False
    
    def _calculate_total_distance(self):
        total_dist = 0
        for crystal in self.crystals:
            target_pos = self.targets[crystal["id"]]["pos"]
            dist = abs(crystal["pos"][0] - target_pos[0]) + abs(crystal["pos"][1] - target_pos[1])
            total_dist += dist
        return total_dist

    def _check_level_complete(self):
        return self._calculate_total_distance() == 0

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level_index + 1,
            "moves_remaining": self.moves_remaining,
        }

    def _iso_to_screen(self, gx, gy, z=0):
        screen_x = self.screen.get_width() / 2 + (gx - gy) * self.TILE_WIDTH_HALF
        screen_y = 100 + (gx + gy) * self.TILE_HEIGHT_HALF - z
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, gx, gy, color_pack, alpha=255):
        base_color, top_color, side_color = color_pack
        
        x, y = self._iso_to_screen(gx, gy)
        h = self.CRYSTAL_HEIGHT
        w = self.TILE_WIDTH_HALF
        th = self.TILE_HEIGHT_HALF

        points_top = [ (x, y - h), (x + w, y - h + th), (x, y - h + 2 * th), (x - w, y - h + th) ]
        points_left = [ (x - w, y - h + th), (x, y - h + 2 * th), (x, y + 2 * th), (x - w, y + th) ]
        points_right = [ (x + w, y - h + th), (x, y - h + 2 * th), (x, y + 2 * th), (x + w, y + th) ]

        # Apply alpha
        top_color_a = (*top_color, alpha)
        side_color_a = (*side_color, alpha)
        
        pygame.gfxdraw.filled_polygon(surface, points_top, top_color_a)
        pygame.gfxdraw.aapolygon(surface, points_top, top_color_a)
        pygame.gfxdraw.filled_polygon(surface, points_left, side_color_a)
        pygame.gfxdraw.aapolygon(surface, points_left, side_color_a)
        pygame.gfxdraw.filled_polygon(surface, points_right, side_color_a)
        pygame.gfxdraw.aapolygon(surface, points_right, side_color_a)

    def _draw_iso_tile(self, surface, gx, gy, color):
        x, y = self._iso_to_screen(gx, gy)
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        points = [(x, y), (x + w, y + h), (x, y + 2 * h), (x - w, y + h)]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        pygame.gfxdraw.aapolygon(surface, points, (color[0]+20, color[1]+20, color[2]+20))

    def _render_game(self):
        # Draw grid
        for r in range(self.grid_size[1]):
            for c in range(self.grid_size[0]):
                self._draw_iso_tile(self.screen, c, r, self.COLOR_GRID)

        # Draw targets (ghosts)
        for target in self.targets:
            color_pack = self.CRYSTAL_COLORS[self.crystals[target["id"]]["type"]]
            self._draw_iso_cube(self.screen, target["pos"][0], target["pos"][1], color_pack, alpha=50)

        # Draw crystals
        for i, crystal in enumerate(self.crystals):
            color_pack = self.CRYSTAL_COLORS[crystal["type"]]
            self._draw_iso_cube(self.screen, crystal["pos"][0], crystal["pos"][1], color_pack)
            # Add glow if selected
            if self.selected_crystal_idx == i:
                pulse = (math.sin(self.steps * 0.3) + 1) / 2
                glow_size = int(pulse * 4)
                x, y = self._iso_to_screen(crystal["pos"][0], crystal["pos"][1], z=self.CRYSTAL_HEIGHT/2)
                pygame.gfxdraw.filled_circle(self.screen, x, y, 18 + glow_size, (*self.COLOR_CURSOR, 50))
                pygame.gfxdraw.aacircle(self.screen, x, y, 18 + glow_size, (*self.COLOR_CURSOR, 100))

        # Draw cursor and selection indicators
        cursor_x, cursor_y = self.cursor_pos
        cursor_color = self.COLOR_CURSOR
        if self.game_phase == 'DIRECT':
            cursor_color = self.CRYSTAL_COLORS[self.crystals[self.selected_crystal_idx]['type']][0]

        x, y = self._iso_to_screen(cursor_x, cursor_y)
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        points = [(x, y), (x + w, y + h), (x, y + 2 * h), (x - w, y + h)]
        pygame.draw.lines(self.screen, cursor_color, True, points, 2)

    def _render_ui(self):
        # Top-left: Moves
        moves_text = self.font_m.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        # Top-right: Level
        level_text = self.font_m.render(f"Level: {self.current_level_index + 1} / {len(self.levels)}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.screen.get_width() - level_text.get_width() - 20, 20))

        # Bottom: Score
        score_text = self.font_s.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, self.screen.get_height() - score_text.get_height() - 10))

        # Center message for level complete/win/loss
        if self.level_complete_flash > 0:
            self.level_complete_flash -= 1
            text = self.font_l.render("LEVEL COMPLETE", True, self.COLOR_SUCCESS)
            text_rect = text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text, text_rect)
        
        if self.game_over:
            if self.game_won_flash > 0:
                self.game_won_flash -= 1
                msg = "YOU WIN!"
                color = self.COLOR_SUCCESS
            else:
                msg = "GAME OVER"
                color = self.COLOR_FAIL
            
            text = self.font_l.render(msg, True, color)
            text_rect = text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(text, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # This block allows you to play the game manually for testing
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use 'dummy' for headless, 'x11' or 'windows' for visible

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen_width, screen_height = 640, 400
    display_screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Crystal Caverns")
    
    terminated = False
    clock = pygame.time.Clock()
    
    print(env.user_guide)

    while not terminated:
        movement, space, shift = 0, 0, 0
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        # The game only advances on an action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Level: {info['level']}, Moves: {info['moves_remaining']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Since auto_advance is False, we need to control the loop rate
        # A small delay prevents the loop from consuming 100% CPU
        # and makes manual play responsive.
        clock.tick(30) 

    print(f"Game Over! Final Score: {info['score']}")
    pygame.time.wait(2000)
    env.close()