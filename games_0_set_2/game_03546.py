
# Generated: 2025-08-27T23:40:35.455780
# Source Brief: brief_03546.md
# Brief Index: 3546

        
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
        "Controls: Arrow keys to push all crystals in a cardinal direction. "
        "The goal is to move crystals next to their matching color path triggers."
    )

    game_description = (
        "An isometric puzzle game. Push colored crystals around a cavern to illuminate all five "
        "matching paths before the 60-second timer runs out. Plan your moves carefully!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 30, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 20)

        # Game Constants
        self.GRID_WIDTH = 12
        self.GRID_HEIGHT = 8
        self.TILE_WIDTH_HALF = 32
        self.TILE_HEIGHT_HALF = 16
        self.ISO_OFFSET_X = 640 // 2
        self.ISO_OFFSET_Y = 400 // 2 - self.GRID_HEIGHT * self.TILE_HEIGHT_HALF

        self.TIME_LIMIT_SECONDS = 60
        self.FPS = 30
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.ACTION_COOLDOWN_FRAMES = 8 # Gives moves a sense of weight

        # Colors
        self.COLOR_BG = (15, 18, 42)
        self.COLOR_WALL = (40, 45, 80)
        self.COLOR_FLOOR = (25, 30, 60)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.CRYSTAL_COLORS = {
            "red": ((255, 80, 80), (200, 50, 50), (150, 30, 30)),
            "green": ((80, 255, 80), (50, 200, 50), (30, 150, 30)),
            "blue": ((80, 80, 255), (50, 50, 200), (30, 30, 150)),
            "yellow": ((255, 255, 80), (200, 200, 50), (150, 150, 30)),
            "purple": ((200, 80, 255), (150, 50, 200), (100, 30, 150)),
        }
        self.PATH_COLOR_UNLIT = (60, 60, 70)

        # Initialize state variables
        self.crystals = []
        self.paths = []
        self.particles = []
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left = 0
        self.action_cooldown = 0
        self.last_lit_count = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        self.action_cooldown = 0
        self.last_lit_count = 0
        self.particles = []

        self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]
        
        self.steps += 1
        self.time_left -= 1
        if self.action_cooldown > 0:
            self.action_cooldown -= 1

        reward = -0.01  # Time penalty

        # Process movement if cooldown is over and it's not a no-op
        if movement != 0 and self.action_cooldown == 0 and not self.game_over:
            self._push_crystals(movement)
            self.action_cooldown = self.ACTION_COOLDOWN_FRAMES
            # sfx: crystal_slide

        lit_count, newly_lit = self._update_paths()
        
        # Calculate reward
        if newly_lit > 0:
            reward += newly_lit * 5.0
            # sfx: path_lit
        reward += lit_count * 0.1 # Continuous reward for lit paths

        self.last_lit_count = lit_count
        self.score += reward

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if lit_count == len(self.paths):
                self.score += 100.0 # Win bonus
                # sfx: win_jingle
            else:
                self.score -= 100.0 # Loss penalty
                # sfx: loss_buzzer

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "time_left": self.time_left / self.FPS,
            "paths_lit": self.last_lit_count,
        }
    
    # --- Game Logic ---

    def _generate_puzzle(self):
        self.crystals = []
        self.paths = []
        
        # Create a grid to track occupied spaces
        grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Place paths
        colors = list(self.CRYSTAL_COLORS.keys())
        for i, color_name in enumerate(colors):
            while True:
                px, py = self.np_random.integers(1, self.GRID_WIDTH - 1), self.np_random.integers(1, self.GRID_HEIGHT - 1)
                if grid[px, py] == 0:
                    self.paths.append({"pos": (px, py), "color_name": color_name, "lit": False})
                    grid[px, py] = 1 # Mark as occupied by a path
                    break
        
        # Place crystals on top of paths (solved state)
        for path in self.paths:
            px, py = path["pos"]
            self.crystals.append({"pos": (px, py), "color_name": path["color_name"]})
            grid[px,py] = 2 # Mark as occupied by a crystal

        # Scramble the puzzle by applying random reverse pushes
        for _ in range(self.np_random.integers(15, 25)):
            # 1=up, 2=down, 3=left, 4=right
            # We push in the opposite direction of the scramble move
            scramble_move = self.np_random.integers(1, 5)
            self._push_crystals(scramble_move, scramble=True)

    def _push_crystals(self, movement, scramble=False):
        # 1=up (-y), 2=down (+y), 3=left (-x), 4=right (+x)
        direction_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = direction_map[movement]

        # Sort crystals to handle chain reactions correctly
        # Iterate from the direction of the push
        sort_key = lambda c: c["pos"][0] * dx + c["pos"][1] * dy
        sorted_crystals = sorted(self.crystals, key=sort_key, reverse=True)
        
        for crystal in sorted_crystals:
            cx, cy = crystal["pos"]
            
            # Find destination
            while True:
                nx, ny = cx + dx, cy + dy
                
                # Check wall collision
                if not (0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT):
                    break
                
                # Check other crystal collision
                is_blocked = any(c["pos"] == (nx, ny) for c in self.crystals)
                if is_blocked:
                    break
                
                cx, cy = nx, ny
            
            crystal["pos"] = (cx, cy)

    def _update_paths(self):
        lit_count = 0
        newly_lit = 0
        
        crystal_positions = {c["pos"]: c["color_name"] for c in self.crystals}

        for path in self.paths:
            was_lit = path["lit"]
            path["lit"] = False
            px, py = path["pos"]
            
            # Check adjacent squares
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_pos = (px + dx, py + dy)
                if crystal_positions.get(check_pos) == path["color_name"]:
                    path["lit"] = True
                    break
            
            if path["lit"]:
                lit_count += 1
                if not was_lit:
                    newly_lit += 1
                    screen_pos = self._iso_to_screen(px, py)
                    self._spawn_particles(screen_pos, self.CRYSTAL_COLORS[path["color_name"]][0])

        return lit_count, newly_lit

    def _check_termination(self):
        if self.time_left <= 0:
            return True
        if self.last_lit_count == len(self.paths):
            return True
        return False

    # --- Rendering ---

    def _iso_to_screen(self, gx, gy):
        sx = self.ISO_OFFSET_X + (gx - gy) * self.TILE_WIDTH_HALF
        sy = self.ISO_OFFSET_Y + (gx + gy) * self.TILE_HEIGHT_HALF
        return int(sx), int(sy)

    def _draw_iso_cube(self, pos, color_tuple):
        x, y = pos
        top_color, left_color, right_color = color_tuple
        
        # Points for the cube
        p_top = (x, y)
        p_right = (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
        p_bottom = (x, y + self.TILE_HEIGHT_HALF * 2)
        p_left = (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
        p_top_face = (p_left, p_top, p_right, (x, y + self.TILE_HEIGHT_HALF))

        # Draw faces
        pygame.gfxdraw.filled_polygon(self.screen, [p_top_face[0], p_top_face[3], p_bottom, p_left], left_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_top_face[3], p_right, p_bottom, p_top_face[3]], right_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_left, p_top, p_right, p_top_face[3]], top_color)
        # Draw outlines for clarity
        pygame.draw.polygon(self.screen, (0,0,0,50), [p_left, p_top, p_right, p_top_face[3]], 2)
        pygame.draw.line(self.screen, (0,0,0,50), p_top_face[3], p_bottom, 2)

    def _render_game(self):
        # Draw floor and walls
        for y in range(self.GRID_HEIGHT + 2):
            for x in range(self.GRID_WIDTH + 2):
                is_wall = not (1 <= x <= self.GRID_WIDTH and 1 <= y <= self.GRID_HEIGHT)
                color = self.COLOR_WALL if is_wall else self.COLOR_FLOOR
                
                screen_pos = self._iso_to_screen(x - 1, y - 1)
                tile_points = [
                    (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_HALF),
                    (screen_pos[0] + self.TILE_WIDTH_HALF, screen_pos[1]),
                    (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_HALF),
                    (screen_pos[0] - self.TILE_WIDTH_HALF, screen_pos[1]),
                ]
                pygame.gfxdraw.filled_polygon(self.screen, tile_points, color)
                pygame.gfxdraw.aapolygon(self.screen, tile_points, (0,0,0,20))

        # Sort paths and crystals for correct draw order
        render_queue = []
        for path in self.paths:
            render_queue.append(('path', path))
        for crystal in self.crystals:
            render_queue.append(('crystal', crystal))
        
        render_queue.sort(key=lambda item: (item[1]['pos'][0] + item[1]['pos'][1], item[0] == 'crystal'))

        for item_type, item_data in render_queue:
            if item_type == 'path':
                px, py = item_data['pos']
                screen_pos = self._iso_to_screen(px, py)
                color_name = item_data['color_name']
                
                tile_points = [
                    (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_HALF),
                    (screen_pos[0] + self.TILE_WIDTH_HALF, screen_pos[1]),
                    (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_HALF),
                    (screen_pos[0] - self.TILE_WIDTH_HALF, screen_pos[1]),
                ]

                if item_data['lit']:
                    color = self.CRYSTAL_COLORS[color_name][0]
                    pygame.gfxdraw.filled_polygon(self.screen, tile_points, color)
                    # Glowing effect
                    glow_color = (*color, 60)
                    for i in range(3, 8, 2):
                        pygame.gfxdraw.aapolygon(self.screen, [
                            (p[0] - i, p[1]), (p[0], p[1] - i),
                            (p[0] + i, p[1]), (p[0], p[1] + i)
                        ], glow_color)
                else:
                    pygame.gfxdraw.filled_polygon(self.screen, tile_points, self.PATH_COLOR_UNLIT)
                
                # Draw a smaller inner polygon to mark the trigger
                inner_points = [
                    (screen_pos[0], screen_pos[1] + self.TILE_HEIGHT_HALF // 2),
                    (screen_pos[0] + self.TILE_WIDTH_HALF // 2, screen_pos[1]),
                    (screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_HALF // 2),
                    (screen_pos[0] - self.TILE_WIDTH_HALF // 2, screen_pos[1]),
                ]
                outline_color = self.CRYSTAL_COLORS[color_name][1]
                pygame.gfxdraw.aapolygon(self.screen, inner_points, outline_color)


            elif item_type == 'crystal':
                cx, cy = item_data['pos']
                screen_pos = self._iso_to_screen(cx, cy)
                color_tuple = self.CRYSTAL_COLORS[item_data['color_name']]
                self._draw_iso_cube((screen_pos[0], screen_pos[1] - self.TILE_HEIGHT_HALF), color_tuple)
        
        self._update_and_draw_particles()

    def _render_ui(self):
        # Paths Lit Counter
        paths_text = f"Paths: {self.last_lit_count} / {len(self.paths)}"
        text_surf = self.font_small.render(paths_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (15, 15))

        # Timer
        time_str = f"Time: {max(0, self.time_left / self.FPS):.1f}"
        text_surf = self.font_small.render(time_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.screen.get_width() - text_surf.get_width() - 15, 15))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.last_lit_count == len(self.paths):
                msg = "SUCCESS!"
            else:
                msg = "TIME UP!"
            
            text_surf = self.font_large.render(msg, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(640 / 2, 400 / 2))
            self.screen.blit(text_surf, text_rect)

    # --- Particle System ---
    def _spawn_particles(self, pos, color):
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": random.randint(15, 30),
                "color": color
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
            else:
                size = max(1, p["life"] / 6)
                pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(size))

    # --- Validation ---
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
    
    # --- Manual Play Loop ---
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # Human controls
        keys = pygame.key.get_pressed()
        action.fill(0)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action[0] = 1 # up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action[0] = 2 # down
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3 # left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4 # right
        
        if keys[pygame.K_r]: # Reset
             obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    pygame.quit()