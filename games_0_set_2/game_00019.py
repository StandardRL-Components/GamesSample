
# Generated: 2025-08-27T16:22:25.943520
# Source Brief: brief_00019.md
# Brief Index: 19

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move the cursor. Press Shift to cycle crystal types. "
        "Press Space to place the selected crystal."
    )

    game_description = (
        "Navigate a procedurally generated crystal cavern by strategically placing crystals "
        "to redirect a laser beam and guide it to the exit portal."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.FONT_UI = pygame.font.Font(None, 24)
        self.FONT_MSG = pygame.font.Font(None, 50)

        # --- Visual & World Design ---
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_WALL = (60, 60, 80)
        self.COLOR_WALL_GLOW = (80, 80, 100)
        self.COLOR_PLAYER = (255, 255, 0)
        self.COLOR_PLAYER_GLOW = (255, 255, 100, 100)
        self.COLOR_EXIT = (0, 255, 128)
        self.COLOR_EXIT_GLOW = (100, 255, 200, 100)
        self.COLOR_LASER_OUTER = (255, 0, 100, 100)
        self.COLOR_LASER_INNER = (255, 200, 220)
        self.COLOR_CURSOR = (255, 255, 255)
        self.CRYSTAL_COLORS = [
            ((0, 150, 255), (100, 200, 255, 100)),  # Blue: 45-degree
            ((255, 150, 0), (255, 200, 100, 100)),  # Yellow: 90-degree
            ((200, 0, 255), (220, 100, 255, 100)),  # Purple: Splitter
        ]
        self.CRYSTAL_NAMES = ["DEFLECTOR", "REFLECTOR", "SPLITTER"]

        # --- Grid & Isometric Projection ---
        self.grid_w, self.grid_h = 22, 16
        self.tile_w, self.tile_h = 28, 14
        self.origin_x = self.screen_width // 2
        self.origin_y = 80

        # --- Game State (Persistent) ---
        self.successful_episodes = 0
        self.wall_count = 5

        # --- Game State (Per Episode) ---
        self.player_pos = (0, 0)
        self.exit_pos = (0, 0)
        self.laser_origin = (0, 0)
        self.laser_origin_dir = (0, 0)
        self.walls = set()
        self.placed_crystals = {}
        self.crystal_counts = {}
        self.selected_crystal_type = 0
        self.cursor_pos = (0, 0)
        self.laser_paths = []
        self.laser_hit_info = {"type": None, "pos": None}
        self.last_laser_dist_to_exit = float('inf')
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.timer = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []

        self.reset()
        self.validate_implementation()

    def _grid_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * (self.tile_w / 2)
        iso_y = self.origin_y + (x + y) * (self.tile_h / 2)
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, x, y, color, glow_color=None, height_mod=0):
        iso_x, iso_y = self._grid_to_iso(x, y)
        iso_y -= height_mod * self.tile_h

        if glow_color:
            glow_radius = int(self.tile_w * 0.8)
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
            surface.blit(s, (iso_x - glow_radius, iso_y - glow_radius))

        points = [
            (iso_x, iso_y - self.tile_h),
            (iso_x + self.tile_w / 2, iso_y - self.tile_h / 2),
            (iso_x, iso_y),
            (iso_x - self.tile_w / 2, iso_y - self.tile_h / 2),
        ]
        pygame.draw.polygon(surface, [c * 0.8 for c in color], points)
        pygame.draw.polygon(surface, (0, 0, 0), points, 1)

    def _generate_cavern(self):
        self.walls.clear()
        valid_positions = set((x, y) for x in range(self.grid_w) for y in range(self.grid_h))

        # Generate walls around the border
        for i in range(self.grid_w):
            self.walls.add((i, -1))
            self.walls.add((i, self.grid_h))
        for i in range(self.grid_h):
            self.walls.add((-1, i))
            self.walls.add((self.grid_w, i))

        # Generate internal walls
        for _ in range(self.wall_count):
            x, y = self.np_random.integers(0, self.grid_w), self.np_random.integers(0, self.grid_h)
            length = self.np_random.integers(3, 7)
            direction = self.np_random.choice([ (1,0), (0,1) ])
            for i in range(length):
                pos = (x + i * direction[0], y + i * direction[1])
                if 0 <= pos[0] < self.grid_w and 0 <= pos[1] < self.grid_h:
                    self.walls.add(pos)
        
        valid_positions -= self.walls

        # Place game elements
        positions = self.np_random.choice(list(valid_positions), size=4, replace=False)
        self.player_pos = tuple(positions[0])
        self.exit_pos = tuple(positions[1])
        self.laser_origin = tuple(positions[2])
        self.cursor_pos = tuple(positions[3])

        # Ensure laser doesn't start pointing at a wall
        possible_dirs = [(1,0), (-1,0), (0,1), (0,-1)]
        self.np_random.shuffle(possible_dirs)
        self.laser_origin_dir = possible_dirs[0]
        for d in possible_dirs:
            if (self.laser_origin[0] + d[0], self.laser_origin[1] + d[1]) not in self.walls:
                self.laser_origin_dir = d
                break

    def _trace_laser(self):
        beams = [(self.laser_origin, self.laser_origin_dir)]
        paths = []
        hit_info = {"type": None, "pos": None}
        closest_dist = float('inf')
        
        for _ in range(30): # Max reflections to prevent infinite loops
            if not beams:
                break
            
            next_beams = []
            for start_pos, direction in beams:
                path_segment = [start_pos]
                current_pos = start_pos
                
                for _ in range(max(self.grid_w, self.grid_h) * 2):
                    next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                    
                    if next_pos == self.player_pos:
                        hit_info = {"type": "PLAYER", "pos": next_pos}
                        path_segment.append(next_pos)
                        break
                    if next_pos == self.exit_pos:
                        hit_info = {"type": "EXIT", "pos": next_pos}
                        path_segment.append(next_pos)
                        break
                    if next_pos in self.walls:
                        hit_info = {"type": "WALL", "pos": next_pos}
                        path_segment.append(next_pos)
                        # Reflection
                        if direction[0] != 0: # Horizontal beam hits vertical wall
                            next_beams.append((current_pos, (-direction[0], direction[1])))
                        else: # Vertical beam hits horizontal wall
                            next_beams.append((current_pos, (direction[0], -direction[1])))
                        break
                    if next_pos in self.placed_crystals:
                        hit_info = {"type": "CRYSTAL", "pos": next_pos}
                        path_segment.append(next_pos)
                        crystal_type = self.placed_crystals[next_pos]
                        dx, dy = direction
                        if crystal_type == 0: # Blue: 45-degree
                            # Simple cross product logic for 45-deg turns
                            next_beams.append((current_pos, (dy, dx)))
                        elif crystal_type == 1: # Yellow: 90-degree
                            next_beams.append((current_pos, (-dx, -dy)))
                        elif crystal_type == 2: # Purple: Splitter
                            next_beams.append((current_pos, (dy, dx)))
                            next_beams.append((current_pos, (-dy, -dx)))
                        break
                        
                    current_pos = next_pos
                
                paths.append(path_segment)
                if hit_info["type"] in ["PLAYER", "EXIT"]:
                    beams = [] # Terminate all tracing
                    break
                
                dist = math.hypot(current_pos[0] - self.exit_pos[0], current_pos[1] - self.exit_pos[1])
                closest_dist = min(closest_dist, dist)

            beams = next_beams
            
        return paths, hit_info, closest_dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over and self.win_message == "LEVEL CLEAR":
            self.successful_episodes += 1
            if self.successful_episodes > 0 and self.successful_episodes % 5 == 0:
                self.wall_count = min(20, self.wall_count + 1)

        self._generate_cavern()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.timer = 60
        self.placed_crystals.clear()
        self.crystal_counts = {0: 3, 1: 3, 2: 2} # Blue, Yellow, Purple
        self.selected_crystal_type = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles.clear()

        self.laser_paths, self.laser_hit_info, self.last_laser_dist_to_exit = self._trace_laser()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        self.game_over = False
        
        # --- Handle Actions ---
        dx, dy = 0, 0
        if movement == 1: dy = -1 # Up
        elif movement == 2: dy = 1 # Down
        elif movement == 3: dx = -1 # Left
        elif movement == 4: dx = 1 # Right
        
        self.cursor_pos = (
            max(0, min(self.grid_w - 1, self.cursor_pos[0] + dx)),
            max(0, min(self.grid_h - 1, self.cursor_pos[1] + dy))
        )

        # Cycle crystal on shift press
        if shift_held and not self.prev_shift_held:
            self.selected_crystal_type = (self.selected_crystal_type + 1) % len(self.CRYSTAL_COLORS)
            # sfx: crystal_cycle.wav

        # Place crystal on space press
        crystal_placed = False
        purple_used = False
        if space_held and not self.prev_space_held:
            can_place = (
                self.cursor_pos not in self.walls and
                self.cursor_pos not in self.placed_crystals and
                self.cursor_pos != self.player_pos and
                self.cursor_pos != self.exit_pos and
                self.crystal_counts[self.selected_crystal_type] > 0
            )
            if can_place:
                self.placed_crystals[self.cursor_pos] = self.selected_crystal_type
                self.crystal_counts[self.selected_crystal_type] -= 1
                crystal_placed = True
                if self.selected_crystal_type == 2:
                    purple_used = True
                    reward += 5  # Event-based reward for using splitter
                # sfx: crystal_place.wav
        
        self.prev_space_held, self.prev_shift_held = space_held, shift_held
        
        # --- Update Game State ---
        self.steps += 1
        self.timer -= 1
        
        if crystal_placed:
            self.laser_paths, self.laser_hit_info, new_dist = self._trace_laser()
            dist_change = self.last_laser_dist_to_exit - new_dist
            if abs(dist_change) > 0.01:
                reward += np.sign(dist_change) * 0.1
            self.last_laser_dist_to_exit = new_dist
        
        # --- Check Termination & Final Rewards ---
        hit_type = self.laser_hit_info["type"]
        if hit_type == "EXIT":
            self.game_over = True
            self.win_message = "LEVEL CLEAR"
            reward += 100
            # sfx: win.wav
        elif hit_type == "PLAYER":
            self.game_over = True
            self.win_message = "PLAYER HIT"
            reward -= 100
            # sfx: lose.wav
        elif self.timer <= 0:
            self.game_over = True
            self.win_message = "TIME UP"
            reward -= 10
            # sfx: timeout.wav
        
        if self.steps >= 1000:
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _render_game(self):
        # Draw grid floor
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                if (c,r) not in self.walls:
                    iso_x, iso_y = self._grid_to_iso(c, r)
                    pygame.gfxdraw.filled_polygon(self.screen, [
                        (iso_x, iso_y), (iso_x + self.tile_w / 2, iso_y + self.tile_h / 2),
                        (iso_x, iso_y + self.tile_h), (iso_x - self.tile_w / 2, iso_y + self.tile_h / 2)
                    ], (30, 25, 45))

        # Draw walls
        for x, y in self.walls:
            if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                self._draw_iso_cube(self.screen, x, y, self.COLOR_WALL, self.COLOR_WALL_GLOW)

        # Draw exit and player
        self._draw_iso_cube(self.screen, self.exit_pos[0], self.exit_pos[1], self.COLOR_EXIT, self.COLOR_EXIT_GLOW, height_mod=-0.5)
        self._draw_iso_cube(self.screen, self.player_pos[0], self.player_pos[1], self.COLOR_PLAYER, self.COLOR_PLAYER_GLOW)
        
        # Draw placed crystals
        for pos, type_idx in self.placed_crystals.items():
            color, glow = self.CRYSTAL_COLORS[type_idx]
            self._draw_iso_cube(self.screen, pos[0], pos[1], color, glow)
        
        # Draw laser
        for path in self.laser_paths:
            if len(path) > 1:
                iso_path = [self._grid_to_iso(p[0], p[1]) for p in path]
                pygame.draw.lines(self.screen, self.COLOR_LASER_OUTER, False, iso_path, 8)
                pygame.draw.lines(self.screen, self.COLOR_LASER_INNER, False, iso_path, 2)
                # sfx: laser_hum.wav (loop)
                # Add hit particles
                if self.laser_hit_info["pos"]:
                    hit_pos_iso = self._grid_to_iso(*iso_path[-1])
                    if self.np_random.random() < 0.5:
                        for _ in range(5):
                            self.particles.append([
                                list(hit_pos_iso), 
                                [self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)],
                                self.np_random.integers(5, 15),
                                self.COLOR_LASER_INNER
                            ])

        # Draw cursor
        cx, cy = self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1])
        cursor_points = [
            (cx, cy - self.tile_h / 2), (cx + self.tile_w / 2, cy),
            (cx, cy + self.tile_h / 2), (cx - self.tile_w / 2, cy)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, cursor_points, 2)
        
        # Update and draw particles
        for p in self.particles[:]:
            p[0][0] += p[1][0]
            p[0][1] += p[1][1]
            p[2] -= 0.5
            if p[2] <= 0:
                self.particles.remove(p)
            else:
                pygame.draw.circle(self.screen, p[3], p[0], int(p[2]))

    def _render_ui(self):
        # Timer
        timer_text = self.FONT_UI.render(f"ACTIONS LEFT: {self.timer}", True, (255, 255, 255))
        self.screen.blit(timer_text, (self.screen_width - timer_text.get_width() - 10, 10))
        
        # Score
        score_text = self.FONT_UI.render(f"SCORE: {self.score:.1f}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Crystal selection UI
        ui_box_rect = pygame.Rect(10, self.screen_height - 60, 250, 50)
        pygame.draw.rect(self.screen, (20, 20, 30, 200), ui_box_rect, border_radius=5)
        
        for i in range(len(self.CRYSTAL_COLORS)):
            x_offset = 25 + i * 80
            color, _ = self.CRYSTAL_COLORS[i]
            
            # Draw icon
            icon_rect = pygame.Rect(x_offset, ui_box_rect.y + 15, 20, 20)
            pygame.draw.rect(self.screen, color, icon_rect)
            
            # Draw count
            count_text = self.FONT_UI.render(f"x{self.crystal_counts[i]}", True, (255, 255, 255))
            self.screen.blit(count_text, (icon_rect.right + 5, icon_rect.y))
            
            # Highlight selected
            if i == self.selected_crystal_type:
                pygame.draw.rect(self.screen, (255, 255, 0), (x_offset - 5, ui_box_rect.y + 5, 70, 40), 2, border_radius=5)
        
        # Game Over Message
        if self.game_over and self.win_message:
            overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg_text = self.FONT_MSG.render(self.win_message, True, (255, 255, 255))
            self.screen.blit(msg_text, msg_text.get_rect(center=self.screen.get_rect().center))

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
            "timer": self.timer,
            "cursor_pos": self.cursor_pos,
            "selected_crystal": self.CRYSTAL_NAMES[self.selected_crystal_type],
            "laser_hit": self.laser_hit_info["type"]
        }
        
    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        print("🔬 Validating implementation...")
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the game and play it manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Override screen for display
    env.screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Crystal Cavern")
    
    terminated = False
    action = [0, 0, 0] # No-op, release, release

    print("\n" + "="*30)
    print("CRYSTAL CAVERN - MANUAL PLAY")
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
        keys = pygame.key.get_pressed()
        
        # Map keys to MultiDiscrete action
        mov = 0 # none
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        env.screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if info['steps'] > 0 and reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.1f}, Reward: {reward:.1f}, Timer: {info['timer']}")

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.1f}")
            pygame.time.wait(3000) # Pause for 3 seconds on game over
            obs, info = env.reset()
            terminated = False

    env.close()