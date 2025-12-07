# Generated: 2025-08-27T20:02:08.462049
# Source Brief: brief_02322.md
# Brief Index: 2322

        
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
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through crystal types. "
        "Press Space to place a crystal at the cursor or remove an existing one."
    )

    game_description = (
        "An isometric puzzle game. Guide a laser from the start to the green exit by placing reflective and splitting crystals. "
        "Manage your laser's energy, which depletes with distance and reflections. Solve the puzzle before you run out of energy or steps!"
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 22, 14
        self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF = 15, 8
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80
        
        self.MAX_STEPS = 500
        self.MAX_ENERGY = 1500
        self.MAX_BEAM_ITERATIONS = 100

        # Colors
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_GRID = (25, 35, 55)
        self.COLOR_WALL_DECOR = (20, 28, 48)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_LASER = (255, 20, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BAR = (40, 50, 80)
        self.COLOR_ENERGY = (0, 150, 255)
        
        self.CRYSTAL_COLORS = {
            "REFLECT_45": (255, 80, 80),  # Red
            "REFLECT_90": (80, 255, 80),  # Green
            "SPLIT": (255, 255, 80),      # Yellow
        }
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_crystal = pygame.font.Font(None, 16)
        
        # Game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.crystals = {}
        self.crystal_types = list(self.CRYSTAL_COLORS.keys())
        self.selected_crystal_idx = 0
        self.laser_beams = []
        self.energy = 0
        self.start_pos = (1, self.GRID_HEIGHT // 2)
        self.start_dir = (1, 0) # E
        self.exit_pos = (self.GRID_WIDTH - 2, self.GRID_HEIGHT // 2)
        self.last_dist_to_exit = float('inf')
        self.last_space_press = False
        self.last_shift_press = False
        self._background_decor = []

        # Reflection/split logic
        self.DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)] # E, NE, N, ...
        self._init_reflection_maps()
        
        self.reset()
        # self.validate_implementation() # Commented out for final submission
    
    def _init_reflection_maps(self):
        self.reflect_45_map = {d: (-d[1], -d[0]) for d in self.DIRECTIONS}
        self.reflect_90_map = {d: (-d[1], d[0]) for d in self.DIRECTIONS} # CCW
        
        self.split_map = {}
        for i, d in enumerate(self.DIRECTIONS):
            d1 = self.DIRECTIONS[(i + 1) % 8]
            d2 = self.DIRECTIONS[(i - 1 + 8) % 8]
            self.split_map[d] = [d1, d2]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.crystals = {}
        self.selected_crystal_idx = 0
        
        self.energy = self.MAX_ENERGY
        self.last_space_press = False
        self.last_shift_press = False

        self._generate_background_decor()
        self._calculate_laser_path()
        self.last_dist_to_exit = self._get_laser_dist_to_exit()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Debounce presses
        space_press = space_held and not self.last_space_press
        shift_press = shift_held and not self.last_shift_press
        self.last_space_press = space_held
        self.last_shift_press = shift_held

        reward = -0.1  # Small cost per step to encourage efficiency
        action_taken = False

        # --- Handle Actions ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        if shift_press:
            self.selected_crystal_idx = (self.selected_crystal_idx + 1) % len(self.crystal_types)
            # Sound: UI_CYCLE.ogg

        if space_press:
            pos = tuple(self.cursor_pos)
            if pos != self.start_pos and pos != self.exit_pos:
                if pos in self.crystals:
                    del self.crystals[pos]
                    action_taken = True
                    # Sound: CRYSTAL_REMOVE.ogg
                else:
                    self.crystals[pos] = self.crystal_types[self.selected_crystal_idx]
                    action_taken = True
                    # Sound: CRYSTAL_PLACE.ogg
        
        # --- Update Game State ---
        if action_taken or movement != 0: # Recalculate path if crystals changed or for reward shaping
            self._calculate_laser_path()
            
            # Reward shaping for getting closer to the exit
            current_dist = self._get_laser_dist_to_exit()
            if current_dist < self.last_dist_to_exit:
                reward += 5.0
            elif current_dist > self.last_dist_to_exit:
                reward -= 2.0
            self.last_dist_to_exit = current_dist

        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        win = self._check_win_condition()
        if win:
            reward += 100
            terminated = True
            self.game_over = True
            # Sound: WIN.ogg
        elif self.energy <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 100
            terminated = True
            self.game_over = True
            # Sound: LOSE.ogg
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_laser_path(self):
        self.energy = self.MAX_ENERGY
        self.laser_beams = []
        
        active_beams = [{"pos": self.start_pos, "dir": self.start_dir, "path": [self.start_pos]}]
        visited_states = set()

        for _ in range(self.MAX_BEAM_ITERATIONS):
            if not active_beams: break

            beam = active_beams.pop(0)
            
            state_tuple = (beam["pos"], beam["dir"])
            if state_tuple in visited_states: continue
            visited_states.add(state_tuple)
            
            next_pos = (beam["pos"][0] + beam["dir"][0], beam["pos"][1] + beam["dir"][1])
            self.energy -= 5 # Travel cost

            # Check for wall collision
            if not (0 <= next_pos[0] < self.GRID_WIDTH and 0 <= next_pos[1] < self.GRID_HEIGHT):
                self.energy -= 10 # Reflection cost
                new_dir_x, new_dir_y = beam["dir"]
                if not (0 <= next_pos[0] < self.GRID_WIDTH): new_dir_x *= -1
                if not (0 <= next_pos[1] < self.GRID_HEIGHT): new_dir_y *= -1
                new_dir = (new_dir_x, new_dir_y)
                beam["path"].append(beam["pos"])
                active_beams.append({"pos": beam["pos"], "dir": new_dir, "path": [beam["pos"]]})
                # Sound: LASER_BOUNCE_WALL.ogg
            
            # Check for crystal collision
            elif next_pos in self.crystals:
                self.energy -= 10
                beam["path"].append(next_pos)
                crystal_type = self.crystals[next_pos]
                
                if crystal_type == "REFLECT_45":
                    active_beams.append({"pos": next_pos, "dir": self.reflect_45_map[beam["dir"]], "path": [next_pos]})
                elif crystal_type == "REFLECT_90":
                    active_beams.append({"pos": next_pos, "dir": self.reflect_90_map[beam["dir"]], "path": [next_pos]})
                elif crystal_type == "SPLIT":
                    self.energy -= 10 # Extra cost for split
                    dirs = self.split_map[beam["dir"]]
                    active_beams.append({"pos": next_pos, "dir": dirs[0], "path": [next_pos]})
                    active_beams.append({"pos": next_pos, "dir": dirs[1], "path": [next_pos]})
                # Sound: LASER_BOUNCE_CRYSTAL.ogg
            
            # Check for exit
            elif next_pos == self.exit_pos:
                beam["path"].append(next_pos)
            
            # Empty space, continue beam
            else:
                beam["pos"] = next_pos
                beam["path"].append(next_pos)
                active_beams.append(beam)

            self.laser_beams.append(beam)
            if self.energy <= 0: break
    
    def _check_win_condition(self):
        for beam in self.laser_beams:
            if beam["path"] and beam["path"][-1] == self.exit_pos:
                return True
        return False

    def _get_laser_dist_to_exit(self):
        min_dist = float('inf')
        endpoints = [b["path"][-1] for b in self.laser_beams if b["path"]]
        if not endpoints:
            return math.hypot(self.start_pos[0] - self.exit_pos[0], self.start_pos[1] - self.exit_pos[1])
        
        for pos in endpoints:
            dist = math.hypot(pos[0] - self.exit_pos[0], pos[1] - self.exit_pos[1])
            min_dist = min(min_dist, dist)
        return min_dist

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._draw_background_decor()
        self._draw_grid()
        
        self._draw_iso_tile(self.start_pos, self.COLOR_LASER, glow=True, size_mod=0.6)
        self._draw_iso_tile(self.exit_pos, self.COLOR_EXIT, glow=True, size_mod=0.8)
        
        for pos, type in self.crystals.items():
            self._draw_crystal(pos, type)

        self._draw_laser()
        self._draw_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "energy": self.energy}

    # --- Helper Rendering Functions ---

    def _world_to_iso(self, x, y):
        iso_x = self.ORIGIN_X + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.ORIGIN_Y + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _generate_background_decor(self):
        self._background_decor = []
        for _ in range(20):
            pos = (random.randint(0, self.GRID_WIDTH-1), random.randint(0, self.GRID_HEIGHT-1))
            size = random.uniform(0.1, 0.4)
            self._background_decor.append((pos, size))

    def _draw_background_decor(self):
        for pos, size in self._background_decor:
            self._draw_iso_tile(pos, self.COLOR_WALL_DECOR, size_mod=size)

    def _draw_grid(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._world_to_iso(x, y)
                p2 = self._world_to_iso(x + 1, y)
                p3 = self._world_to_iso(x + 1, y + 1)
                p4 = self._world_to_iso(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

    def _draw_iso_tile(self, pos, color, glow=False, size_mod=1.0):
        x, y = pos
        w, h = self.TILE_WIDTH_HALF * size_mod, self.TILE_HEIGHT_HALF * size_mod
        center_x, center_y = self._world_to_iso(x + 0.5, y + 0.5)
        
        points = [
            (center_x, center_y - h),
            (center_x + w, center_y),
            (center_x, center_y + h),
            (center_x - w, center_y)
        ]
        
        if glow:
            for i in range(4, 0, -1):
                alpha_color = (*color, 20 * i)
                temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.gfxdraw.filled_polygon(temp_surf, [(int(p[0]), int(p[1])) for p in points], alpha_color)
                self.screen.blit(temp_surf, (0,0))
        
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_crystal(self, pos, type):
        color = self.CRYSTAL_COLORS[type]
        pulse = (math.sin(self.steps * 0.1) + 1) / 2 # 0 to 1
        size = 0.6 + pulse * 0.2
        self._draw_iso_tile(pos, color, glow=True, size_mod=size)
        
    def _draw_laser(self):
        for beam in self.laser_beams:
            if len(beam["path"]) > 1:
                iso_path = [self._world_to_iso(p[0] + 0.5, p[1] + 0.5) for p in beam["path"]]
                self._draw_glowing_line_segments(self.screen, self.COLOR_LASER, iso_path, 3)

    def _draw_glowing_line_segments(self, surf, color, points, width):
        # This function was fixed to correctly implement the intended glow effect.
        # The original code passed an invalid 'blend' argument to pygame.draw.lines.
        # The correct way to achieve an additive blend is to draw on a temporary
        # surface and blit it to the main surface with the BLEND_RGBA_ADD flag.
        
        # Create a temporary surface for drawing. It must support per-pixel alpha.
        temp_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)

        # A constant, low-alpha color is suitable for building up a glow with additive blending.
        glow_color = (*color, 20)

        for i in range(width * 2, 0, -2):
            # Clear the temporary surface for each new line segment to be drawn.
            temp_surf.fill((0, 0, 0, 0))
            
            # Draw the line segment with the specified width on the temporary surface.
            pygame.draw.lines(temp_surf, glow_color, False, points, width=i)
            
            # Blit the line from the temporary surface to the main surface using additive blending.
            surf.blit(temp_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            
        # After drawing the glow, draw the bright, thin core of the laser on top.
        pygame.draw.lines(surf, (255, 255, 255), False, points, width=1)


    def _draw_cursor(self):
        x, y = self.cursor_pos
        w, h = self.TILE_WIDTH_HALF, self.TILE_HEIGHT_HALF
        center_x, center_y = self._world_to_iso(x + 0.5, y + 0.5)
        points = [
            (center_x, center_y - h), (center_x + w, center_y),
            (center_x, center_y + h), (center_x - w, center_y)
        ]
        pygame.draw.lines(self.screen, self.COLOR_CURSOR, True, points, 2)

    def _render_ui(self):
        # Energy Bar
        bar_w, bar_h = 150, 15
        bar_x, bar_y = 10, 10
        energy_ratio = max(0, self.energy / self.MAX_ENERGY)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_ENERGY, (bar_x, bar_y, int(bar_w * energy_ratio), bar_h))
        
        # Time/Steps Left
        time_text = self.font_ui.render(f"STEPS: {self.MAX_STEPS - self.steps}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (bar_x + bar_w + 10, bar_y))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Selected Crystal
        crystal_type = self.crystal_types[self.selected_crystal_idx]
        crystal_color = self.CRYSTAL_COLORS[crystal_type]
        crystal_text = self.font_ui.render("CRYSTAL:", True, self.COLOR_UI_TEXT)
        self.screen.blit(crystal_text, (10, self.HEIGHT - 30))
        pygame.draw.rect(self.screen, crystal_color, (crystal_text.get_width() + 15, self.HEIGHT - 30, 20, 20))
        type_text = self.font_crystal.render(crystal_type.replace("_", " "), True, self.COLOR_UI_TEXT)
        self.screen.blit(type_text, (crystal_text.get_width() + 40, self.HEIGHT - 28))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame Interactive Loop ---
    # This part is for human playtesting and is not part of the Gym interface
    
    # Re-initialize pygame with a display
    pygame.display.init()
    pygame.display.set_caption("Crystal Laser")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    print(env.game_description)

    running = True
    while running:
        # Action mapping from keyboard to MultiDiscrete
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
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            # We only step when a key is pressed, because auto_advance is False
            # For a better human experience, we step on any key press.
            if any(keys):
                obs, reward, terminated, truncated, info = env.step(action)
                if reward != -0.1: # Only print meaningful reward changes
                    print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    pygame.quit()