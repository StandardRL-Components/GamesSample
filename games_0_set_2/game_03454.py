
# Generated: 2025-08-27T23:24:37.170223
# Source Brief: brief_03454.md
# Brief Index: 3454

        
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
        "Controls: Use arrow keys to move the selected robot. "
        "Press SPACE to cycle to the next robot, and SHIFT to cycle to the previous one."
    )

    game_description = (
        "Guide a team of robots through an isometric puzzle environment to collect "
        "scattered parts before the timer runs out."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 20, 15
    TILE_WIDTH_ISO, TILE_HEIGHT_ISO = 40, 20
    NUM_ROBOTS = 3
    NUM_PARTS = 10
    NUM_OBSTACLES = 25
    MAX_TIME_STEPS = 1800  # 60 seconds at 30fps

    MOVE_ANIMATION_FRAMES = 6
    MOVE_ANIMATION_SPEED = 1.0 / MOVE_ANIMATION_FRAMES

    # --- Colors ---
    COLOR_BG = (44, 62, 80)
    COLOR_GRID = (52, 73, 94)
    COLOR_WALL = (41, 52, 65)
    COLOR_WALL_TOP = (52, 63, 77)
    COLOR_ROBOT = (52, 152, 219)
    COLOR_ROBOT_TOP = (82, 182, 249)
    COLOR_PART = (241, 196, 15)
    COLOR_PART_SHINE = (243, 229, 100)
    COLOR_SELECTOR = (46, 204, 113)
    COLOR_SHADOW = (30, 40, 50, 100)
    COLOR_TEXT = (236, 240, 241)

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

        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 28, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.grid_origin_x = self.SCREEN_WIDTH / 2
        self.grid_origin_y = self.SCREEN_HEIGHT / 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_ISO / 3)

        self.robots = []
        self.parts = []
        self.walls = set()
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.game_won = False
        self.selected_robot_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.np_random = None

        self.reset()
        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_TIME_STEPS
        self.game_over = False
        self.game_won = False
        self.selected_robot_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False

        # --- Procedural Generation ---
        valid_spawns = list(
            (r, c) for r in range(self.GRID_HEIGHT) for c in range(self.GRID_WIDTH)
        )

        # Generate walls
        self.walls = set()
        # Create a border of walls
        for r in range(-1, self.GRID_HEIGHT + 1):
            self.walls.add((r, -1))
            self.walls.add((r, self.GRID_WIDTH))
        for c in range(-1, self.GRID_WIDTH + 1):
            self.walls.add((-1, c))
            self.walls.add((self.GRID_HEIGHT, c))
        
        # Add internal obstacles
        internal_spawns = [pos for pos in valid_spawns if 2 < pos[0] < self.GRID_HEIGHT - 2 and 2 < pos[1] < self.GRID_WIDTH - 2]
        if len(internal_spawns) > self.NUM_OBSTACLES:
            obstacle_indices = self.np_random.choice(
                len(internal_spawns), self.NUM_OBSTACLES, replace=False
            )
            for i in obstacle_indices:
                self.walls.add(internal_spawns[i])

        # Filter valid spawns to exclude walls
        valid_spawns = [pos for pos in valid_spawns if pos not in self.walls]

        # Spawn robots and parts
        num_entities_to_spawn = self.NUM_ROBOTS + self.NUM_PARTS
        if len(valid_spawns) < num_entities_to_spawn:
             raise ValueError("Not enough valid spawn points for robots and parts.")

        spawn_indices = self.np_random.choice(
            len(valid_spawns), num_entities_to_spawn, replace=False
        )
        spawn_points = [valid_spawns[i] for i in spawn_indices]

        self.robots = []
        for i in range(self.NUM_ROBOTS):
            pos = spawn_points.pop(0)
            self.robots.append({
                "pos": np.array(pos, dtype=float),
                "target_pos": np.array(pos, dtype=float),
                "anim_progress": 0.0,
            })

        self.parts = []
        for i in range(self.NUM_PARTS):
            pos = spawn_points.pop(0)
            self.parts.append({"pos": pos, "collected": False})

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        self.time_left -= 1
        reward = 0

        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Cycle robots on button press (not hold)
        if space_held and not self.prev_space_held:
            self.selected_robot_index = (self.selected_robot_index + 1) % self.NUM_ROBOTS
            # sfx: UI_CYCLE_SOUND
        if shift_held and not self.prev_shift_held:
            self.selected_robot_index = (self.selected_robot_index - 1 + self.NUM_ROBOTS) % self.NUM_ROBOTS
            # sfx: UI_CYCLE_SOUND

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Move selected robot
        selected_robot = self.robots[self.selected_robot_index]
        if movement != 0 and selected_robot["anim_progress"] == 0.0:
            # 1=up, 2=down, 3=left, 4=right
            # Isometric mapping: up->NE, down->SW, left->NW, right->SE
            move_map = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}
            delta = move_map.get(movement, (0, 0))
            
            # This is a bit confusing, but maps grid movement to iso-view movement
            # In grid space: (row, col)
            # Up (NE in iso): row-1, col
            # Down (SW in iso): row+1, col
            # Left (NW in iso): row, col-1
            # Right (SE in iso): row, col+1
            grid_move_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
            grid_delta = grid_move_map.get(movement, (0,0))
            
            target = tuple(np.round(selected_robot["pos"] + grid_delta).astype(int))
            
            if target not in self.walls:
                selected_robot["target_pos"] = np.array(target, dtype=float)
                selected_robot["anim_progress"] = self.MOVE_ANIMATION_SPEED
                # sfx: ROBOT_MOVE_START_SOUND

        # --- Update Game State ---
        # Animate robots
        for robot in self.robots:
            if robot["anim_progress"] > 0:
                robot["anim_progress"] += self.MOVE_ANIMATION_SPEED
                if robot["anim_progress"] >= 1.0:
                    robot["anim_progress"] = 0.0
                    robot["pos"] = robot["target_pos"]
                    # sfx: ROBOT_MOVE_END_SOUND
                    
                    # Check for part collection after move completes
                    for part in self.parts:
                        if not part["collected"] and tuple(np.round(robot["pos"]).astype(int)) == part["pos"]:
                            part["collected"] = True
                            self.score += 1
                            reward += 0.1
                            # sfx: PART_COLLECT_SOUND

        # --- Check Termination ---
        parts_collected_count = sum(1 for p in self.parts if p["collected"])
        all_parts_collected = parts_collected_count == self.NUM_PARTS
        time_is_up = self.time_left <= 0

        terminated = all_parts_collected or time_is_up
        if terminated and not self.game_over:
            self.game_over = True
            if all_parts_collected:
                self.game_won = True
                reward += 1.0  # Bonus for last part
                reward += 10.0 # Win bonus
                # sfx: GAME_WIN_SOUND
            else: # Time ran out
                reward = -1.0
                # sfx: GAME_LOSE_SOUND

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _iso_to_screen(self, r, c):
        """Converts grid coordinates (row, col) to screen coordinates (x, y)."""
        screen_x = self.grid_origin_x + (c - r) * self.TILE_WIDTH_ISO / 2
        screen_y = self.grid_origin_y + (c + r) * self.TILE_HEIGHT_ISO / 2
        return int(screen_x), int(screen_y)

    def _draw_iso_cube(self, surface, r, c, color, top_color, height=20):
        """Draws a 3D isometric cube at a grid position."""
        x, y = self._iso_to_screen(r, c)
        
        hw, hh = self.TILE_WIDTH_ISO / 2, self.TILE_HEIGHT_ISO / 2
        
        points = [
            (x, y - height),                            # Top
            (x + hw, y + hh - height),                  # Top-right
            (x, y + self.TILE_HEIGHT_ISO - height),     # Top-bottom
            (x - hw, y + hh - height),                  # Top-left
            (x, y + self.TILE_HEIGHT_ISO),              # Bottom
            (x - hw, y + hh),                           # Bottom-left
        ]

        # Draw faces
        pygame.draw.polygon(surface, top_color, [points[0], points[1], points[2], points[3]]) # Top face
        pygame.draw.polygon(surface, color, [points[2], points[4], points[5], points[3]]) # Left face
        # Right face (darker)
        darker_color = tuple(max(0, val - 20) for val in color)
        pygame.draw.polygon(surface, darker_color, [points[1], (points[1][0], points[1][1] + height), points[4], points[2]])

    def _draw_iso_diamond(self, surface, r, c, color, shine_color):
        """Draws a floating, spinning diamond for a part."""
        x, y = self._iso_to_screen(r, c)
        
        # Animate floating and spinning
        float_offset = math.sin(self.steps * 0.1) * 5
        spin_factor = math.cos(self.steps * 0.08)
        
        hw, hh = self.TILE_WIDTH_ISO / 4, self.TILE_HEIGHT_ISO / 4
        
        y_pos = y - 15 - float_offset

        points = [
            (x, y_pos - hh * 1.5),
            (x + hw * spin_factor, y_pos),
            (x, y_pos + hh * 1.5),
            (x - hw * spin_factor, y_pos)
        ]
        
        pygame.draw.polygon(surface, color, points)
        if spin_factor > 0:
            pygame.draw.polygon(surface, shine_color, [points[0], points[1], points[3]])
        else:
            pygame.draw.polygon(surface, shine_color, [points[2], points[1], points[3]])

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)

        # --- Render Grid and Walls ---
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (r,c) in self.walls:
                    self._draw_iso_cube(self.screen, r, c, self.COLOR_WALL, self.COLOR_WALL_TOP, height=20)
                else:
                    x, y = self._iso_to_screen(r, c)
                    hw, hh = self.TILE_WIDTH_ISO / 2, self.TILE_HEIGHT_ISO / 2
                    points = [(x,y), (x+hw, y+hh), (x, y+self.TILE_HEIGHT_ISO), (x-hw, y+hh)]
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)
        
        # --- Render Shadows and Selector ---
        shadow_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        hw, hh = self.TILE_WIDTH_ISO / 2, self.TILE_HEIGHT_ISO / 2
        
        # Draw selector for the active robot
        selected_robot = self.robots[self.selected_robot_index]
        r, c = selected_robot["pos"]
        tr, tc = selected_robot["target_pos"]
        prog = selected_robot["anim_progress"]
        
        interp_r = r + (tr - r) * prog
        interp_c = c + (tc - c) * prog
        
        x, y = self._iso_to_screen(interp_r, interp_c)
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        radius = int(hw * (0.8 + pulse * 0.15))
        selector_color = (*self.COLOR_SELECTOR, int(150 + pulse * 50))
        pygame.gfxdraw.filled_circle(shadow_surface, x, y + int(hh), radius, selector_color)
        pygame.gfxdraw.aacircle(shadow_surface, x, y + int(hh), radius, selector_color)

        # Draw shadows for all robots
        for robot in self.robots:
            r, c = robot["pos"]
            tr, tc = robot["target_pos"]
            prog = robot["anim_progress"]
            interp_r = r + (tr - r) * prog
            interp_c = c + (tc - c) * prog
            x, y = self._iso_to_screen(interp_r, interp_c)
            pygame.gfxdraw.filled_ellipse(shadow_surface, x, y + int(hh), int(hw * 0.8), int(hh * 0.8), self.COLOR_SHADOW)
        
        self.screen.blit(shadow_surface, (0, 0))

        # --- Render Parts ---
        for part in self.parts:
            if not part["collected"]:
                self._draw_iso_diamond(self.screen, *part["pos"], self.COLOR_PART, self.COLOR_PART_SHINE)

        # --- Render Robots ---
        for robot in self.robots:
            r, c = robot["pos"]
            tr, tc = robot["target_pos"]
            prog = robot["anim_progress"]
            
            interp_r = r + (tr - r) * prog
            interp_c = c + (tc - c) * prog

            self._draw_iso_cube(self.screen, interp_r, interp_c, self.COLOR_ROBOT, self.COLOR_ROBOT_TOP)

        # --- Render UI ---
        parts_text = self.font_ui.render(f"Parts: {self.score}/{self.NUM_PARTS}", True, self.COLOR_TEXT)
        self.screen.blit(parts_text, (10, 10))

        time_seconds = self.time_left / 30.0
        time_color = (231, 76, 60) if time_seconds < 10 else self.COLOR_TEXT
        timer_text = self.font_timer.render(f"{time_seconds:.1f}", True, time_color)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 15, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0,0))
            
            msg = "MISSION COMPLETE" if self.game_won else "TIME UP"
            color = (46, 204, 113) if self.game_won else (231, 76, 60)
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "parts_collected": sum(1 for p in self.parts if p["collected"]),
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Robot Collector")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space_held = False
        shift_held = False

        keys = pygame.key.get_pressed()
        if not terminated:
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space_held = True
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_held = True

        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Loop ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press R to reset
                    obs, info = env.reset()
                    terminated = False
        
        # --- Rendering ---
        # The observation is already a rendered frame
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()