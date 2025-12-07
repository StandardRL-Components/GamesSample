import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a wire "
        "start point, then press Space again on a target connection point to connect it."
    )

    game_description = (
        "Repair malfunctioning robots by correctly connecting their internal wiring. "
        "Work fast in this isometric puzzle game before the timer runs out!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 60)

        # --- Game Constants ---
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 120
        self.MAX_STEPS = 1000
        self.CURSOR_SPEED = 8
        self.CLICK_RADIUS = 20

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (30, 35, 50)
        self.COLOR_ROBOT_BODY = (80, 90, 110)
        self.COLOR_ROBOT_SHADOW = (15, 20, 35)
        self.COLOR_POINT = (150, 160, 180)
        self.COLOR_POINT_POTENTIAL = (255, 255, 0) # Yellow
        self.COLOR_WIRE_UNCONNECTED = (180, 180, 200)
        self.COLOR_WIRE_SELECTED = (0, 150, 255) # Blue
        self.COLOR_WIRE_CORRECT = (0, 255, 100) # Green
        self.COLOR_WIRE_INCORRECT_FLASH = (255, 50, 50) # Red
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_BAR_BG = (50, 60, 80)
        self.COLOR_UI_TIMER = (255, 180, 0)

        # --- Isometric Projection Constants ---
        self.ISO_TILE_WIDTH = 32
        self.ISO_TILE_HEIGHT = 16
        self.ISO_ORIGIN_X = self.screen_width // 2
        self.ISO_ORIGIN_Y = 100

        # --- State variables will be initialized in reset() ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.incorrect_moves = 0
        self.cursor_pos = np.array([0, 0], dtype=np.float32)
        self.selected_wire_info = None
        self.previous_space_held = False
        self.particles = []
        self.robots = []

        # --- Validation ---
        # self.validate_implementation() # This can be noisy, call it manually if needed

    def _generate_puzzle(self):
        """Creates the layout of robots and their wiring puzzles."""
        self.robots = []
        robot_positions = [(-4, 4, 0), (4, 4, 0), (-4, -4, 0), (4, -4, 0)]
        
        for i, pos in enumerate(robot_positions):
            # Define connection points relative to the robot's origin
            points = {
                0: (-1, 0, 1.5), 1: (1, 0, 1.5),
                2: (0, -1, 1.5), 3: (0, 1, 1.5),
                4: (-1, 0, 0.5), 5: (1, 0, 0.5),
            }
            # Define the correct connections for this robot's puzzle
            solutions = [(0, 5), (1, 2), (3, 4)]
            # Shuffle solutions for variety, seeded by the environment seed
            # Use a copy to shuffle in-place
            shuffled_solutions = list(solutions)
            self.np_random.shuffle(shuffled_solutions)

            wires = []
            for sol_start, sol_end in shuffled_solutions:
                wires.append({
                    "solution": tuple(sorted((sol_start, sol_end))),
                    "start_point_id": sol_start,
                    "end_point_id": None,
                    "state": "unconnected", # States: unconnected, connected
                    "flash_timer": 0,
                })
            
            robot = {
                "pos": pos,
                "points": points,
                "wires": wires,
                "is_complete": False,
            }
            self.robots.append(robot)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.time_remaining = self.TIME_LIMIT_SECONDS * self.FPS
        self.incorrect_moves = 0
        
        self.cursor_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float32)
        self.selected_wire_info = None # e.g. {'robot_idx': r, 'wire_idx': w}
        self.previous_space_held = False
        
        self.particles = []

        self._generate_puzzle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_remaining -= 1
        reward = 0.0

        # Unpack action
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Update Game Logic ---
        # 1. Cursor Movement
        if movement == 1: self.cursor_pos[1] -= self.CURSOR_SPEED # Up
        elif movement == 2: self.cursor_pos[1] += self.CURSOR_SPEED # Down
        elif movement == 3: self.cursor_pos[0] -= self.CURSOR_SPEED # Left
        elif movement == 4: self.cursor_pos[0] += self.CURSOR_SPEED # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.screen_width)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.screen_height)

        # 2. Handle Action (Space Press)
        is_space_press = space_held and not self.previous_space_held
        if is_space_press:
            clicked_object = self._get_object_under_cursor()
            
            if self.selected_wire_info is None:
                # Try to select a wire
                if clicked_object and clicked_object['type'] == 'wire_start':
                    self.selected_wire_info = {
                        'robot_idx': clicked_object['robot_idx'],
                        'wire_idx': clicked_object['wire_idx'],
                    }
            else:
                # A wire is selected, try to connect it
                sel_info = self.selected_wire_info
                wire = self.robots[sel_info['robot_idx']]['wires'][sel_info['wire_idx']]
                
                if clicked_object and clicked_object['type'] == 'point' and clicked_object['robot_idx'] == sel_info['robot_idx']:
                    # Clicked on a connection point on the same robot
                    attempted_connection = tuple(sorted((wire['start_point_id'], clicked_object['point_id'])))
                    
                    if attempted_connection == wire['solution']:
                        # CORRECT connection
                        wire['state'] = 'connected'
                        wire['end_point_id'] = clicked_object['point_id']
                        reward += 1.0
                        self._create_sparks(clicked_object['pos'], self.COLOR_WIRE_CORRECT)
                        
                        # Check for robot completion
                        if all(w['state'] == 'connected' for w in self.robots[sel_info['robot_idx']]['wires']):
                            self.robots[sel_info['robot_idx']]['is_complete'] = True
                            reward += 10.0
                    else:
                        # INCORRECT connection
                        self.incorrect_moves += 1
                        reward -= 0.1
                        wire['flash_timer'] = int(0.5 * self.FPS) # Flash for 0.5s
                        self._create_sparks(clicked_object['pos'], self.COLOR_WIRE_INCORRECT_FLASH, is_failure=True)
                
                # Deselect wire regardless of outcome
                self.selected_wire_info = None
        
        self.previous_space_held = space_held
        self.score += reward
        
        # --- Check Termination Conditions ---
        all_robots_repaired = all(r['is_complete'] for r in self.robots)
        time_is_up = self.time_remaining <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS
        
        terminated = all_robots_repaired or time_is_up or max_steps_reached
        if terminated and not self.game_over:
            self.game_over = True
            if all_robots_repaired:
                self.score += 100.0
                reward += 100.0
            elif time_is_up:
                self.score -= 100.0
                reward -= 100.0

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        progress = self._calculate_progress()
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": max(0, self.time_remaining / self.FPS),
            "repair_progress_percent": progress,
            "incorrect_moves": self.incorrect_moves,
        }

    def _render_game(self):
        # Draw background grid
        for i in range(-10, 11):
            p1_start, p1_end = self._iso_to_screen(i, -10, 0), self._iso_to_screen(i, 10, 0)
            p2_start, p2_end = self._iso_to_screen(-10, i, 0), self._iso_to_screen(10, i, 0)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p1_start, p1_end)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, p2_start, p2_end)

        # Draw robots, points, and wires
        for r_idx, robot in enumerate(self.robots):
            self._draw_robot_chassis(robot)
            
            # Draw connection points
            for p_id, p_pos in robot['points'].items():
                point_world_pos = (robot['pos'][0] + p_pos[0], robot['pos'][1] + p_pos[1], robot['pos'][2] + p_pos[2])
                point_screen_pos = self._iso_to_screen(*point_world_pos)
                
                is_potential = False
                if self.selected_wire_info and self.selected_wire_info['robot_idx'] == r_idx:
                    is_potential = True
                
                color = self.COLOR_POINT_POTENTIAL if is_potential else self.COLOR_POINT
                pygame.gfxdraw.filled_circle(self.screen, int(point_screen_pos[0]), int(point_screen_pos[1]), 5, color)
                pygame.gfxdraw.aacircle(self.screen, int(point_screen_pos[0]), int(point_screen_pos[1]), 5, color)

            # Draw wires
            for w_idx, wire in enumerate(robot['wires']):
                start_p_local = robot['points'][wire['start_point_id']]
                start_p_world = (robot['pos'][0] + start_p_local[0], robot['pos'][1] + start_p_local[1], robot['pos'][2] + start_p_local[2])
                start_pos = self._iso_to_screen(*start_p_world)

                if wire['state'] == 'connected':
                    end_p_local = robot['points'][wire['end_point_id']]
                    end_p_world = (robot['pos'][0] + end_p_local[0], robot['pos'][1] + end_p_local[1], robot['pos'][2] + end_p_local[2])
                    end_pos = self._iso_to_screen(*end_p_world)
                    color = self.COLOR_WIRE_CORRECT
                else: # Unconnected
                    if self.selected_wire_info and self.selected_wire_info['robot_idx'] == r_idx and self.selected_wire_info['wire_idx'] == w_idx:
                        end_pos = self.cursor_pos
                        color = self.COLOR_WIRE_SELECTED
                    else:
                        end_pos = start_pos
                        color = self.COLOR_WIRE_UNCONNECTED
                
                if wire['flash_timer'] > 0:
                    wire['flash_timer'] -= 1
                    if wire['flash_timer'] % 10 < 5:
                        color = self.COLOR_WIRE_INCORRECT_FLASH

                self._draw_glowing_line(self.screen, color, start_pos, end_pos, 2)
        
        self._update_and_draw_particles()
        self._draw_cursor()

    def _render_ui(self):
        # Timer Bar
        timer_width = self.screen_width - 20
        timer_ratio = max(0, self.time_remaining / (self.TIME_LIMIT_SECONDS * self.FPS))
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (10, 10, timer_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_TIMER, (10, 10, timer_width * timer_ratio, 20))

        # Text Info
        progress = self._calculate_progress()
        progress_text = f"REPAIR: {progress:.0f}%"
        self._draw_text(progress_text, (15, self.screen_height - 35), self.font_main, self.COLOR_UI_TEXT)
        
        errors_text = f"ERRORS: {self.incorrect_moves}"
        self._draw_text(errors_text, (self.screen_width - 120, self.screen_height - 35), self.font_main, self.COLOR_UI_TEXT)

        if self.game_over:
            if all(r['is_complete'] for r in self.robots):
                msg = "VICTORY"
                color = self.COLOR_WIRE_CORRECT
            else:
                msg = "TIME UP"
                color = self.COLOR_WIRE_INCORRECT_FLASH
            self._draw_text(msg, (self.screen_width/2, self.screen_height/2 - 20), self.font_large, color, center=True)


    def _iso_to_screen(self, x, y, z):
        """Converts isometric coordinates to screen coordinates."""
        screen_x = self.ISO_ORIGIN_X + (x - y) * self.ISO_TILE_WIDTH / 2
        screen_y = self.ISO_ORIGIN_Y + (x + y) * self.ISO_TILE_HEIGHT / 2 - z * self.ISO_TILE_HEIGHT
        return int(screen_x), int(screen_y)

    def _draw_robot_chassis(self, robot):
        """Draws a simple isometric box for the robot."""
        x, y, z = robot['pos']
        w, d, h = 2, 2, 1 # width, depth, height in iso units
        
        corners = [
            (x - w/2, y - d/2, z), (x + w/2, y - d/2, z),
            (x + w/2, y + d/2, z), (x - w/2, y + d/2, z),
            (x - w/2, y - d/2, z + h), (x + w/2, y - d/2, z + h),
            (x + w/2, y + d/2, z + h), (x - w/2, y + d/2, z + h),
        ]
        screen_corners = [self._iso_to_screen(*c) for c in corners]

        shadow_points = [screen_corners[0], screen_corners[1], screen_corners[2], screen_corners[3]]
        pygame.gfxdraw.filled_polygon(self.screen, shadow_points, self.COLOR_ROBOT_SHADOW)
        
        top_face = [screen_corners[4], screen_corners[5], screen_corners[6], screen_corners[7]]
        pygame.gfxdraw.filled_polygon(self.screen, top_face, self.COLOR_ROBOT_BODY)
        
        side_face1 = [screen_corners[1], screen_corners[2], screen_corners[6], screen_corners[5]]
        side_face2 = [screen_corners[3], screen_corners[2], screen_corners[6], screen_corners[7]]
        
        # Create slightly darker colors for the side faces to give a 3D effect
        side_color1 = tuple(max(0, c - 10) for c in self.COLOR_ROBOT_BODY)
        side_color2 = tuple(max(0, c - 20) for c in self.COLOR_ROBOT_BODY)
        pygame.gfxdraw.filled_polygon(self.screen, side_face1, side_color1)
        pygame.gfxdraw.filled_polygon(self.screen, side_face2, side_color2)

        pygame.gfxdraw.aapolygon(self.screen, top_face, self.COLOR_GRID)
        pygame.gfxdraw.aapolygon(self.screen, side_face1, self.COLOR_GRID)
        pygame.gfxdraw.aapolygon(self.screen, side_face2, self.COLOR_GRID)
    
    def _draw_glowing_line(self, surface, color, start, end, width):
        """Draws a line with a fake glow effect."""
        glow_color = (*color[:3], 60)
        
        # This requires a surface that supports alpha blending.
        temp_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        pygame.draw.line(temp_surface, glow_color, start, end, width * 4)
        pygame.draw.line(temp_surface, glow_color, start, end, width * 2)
        surface.blit(temp_surface, (0, 0))

        pygame.draw.aaline(surface, color, start, end, True)

    def _draw_cursor(self):
        x, y = int(self.cursor_pos[0]), int(self.cursor_pos[1])
        color = self.COLOR_WIRE_SELECTED if self.selected_wire_info else self.COLOR_UI_TEXT
        pygame.draw.line(self.screen, color, (x - 8, y), (x + 8, y), 2)
        pygame.draw.line(self.screen, color, (x, y - 8), (x, y + 8), 2)
        
    def _draw_text(self, text, pos, font, color, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _get_object_under_cursor(self):
        """Finds the closest clickable object to the cursor."""
        closest_obj = None
        min_dist = self.CLICK_RADIUS

        for r_idx, robot in enumerate(self.robots):
            # Check wire starts (only if a wire isn't already selected)
            if self.selected_wire_info is None:
                for w_idx, wire in enumerate(robot['wires']):
                    if wire['state'] == 'unconnected':
                        p_local = robot['points'][wire['start_point_id']]
                        p_world = (robot['pos'][0] + p_local[0], robot['pos'][1] + p_local[1], robot['pos'][2] + p_local[2])
                        p_screen = self._iso_to_screen(*p_world)
                        dist = np.linalg.norm(self.cursor_pos - p_screen)
                        if dist < min_dist:
                            min_dist = dist
                            closest_obj = {'type': 'wire_start', 'robot_idx': r_idx, 'wire_idx': w_idx}
            
            # Check all connection points (only if a wire is selected)
            if self.selected_wire_info and self.selected_wire_info['robot_idx'] == r_idx:
                for p_id, p_local in robot['points'].items():
                    p_world = (robot['pos'][0] + p_local[0], robot['pos'][1] + p_local[1], robot['pos'][2] + p_local[2])
                    p_screen = self._iso_to_screen(*p_world)
                    dist = np.linalg.norm(self.cursor_pos - p_screen)
                    if dist < min_dist:
                        min_dist = dist
                        closest_obj = {'type': 'point', 'robot_idx': r_idx, 'point_id': p_id, 'pos': p_screen}
        return closest_obj

    def _calculate_progress(self):
        """Calculates the total repair progress percentage."""
        total_wires = sum(len(r['wires']) for r in self.robots)
        if total_wires == 0: return 100
        connected_wires = sum(1 for r in self.robots for w in r['wires'] if w['state'] == 'connected')
        return (connected_wires / total_wires) * 100
    
    def _create_sparks(self, pos, color, count=20, is_failure=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) if not is_failure else self.np_random.uniform(0.5, 2)
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': velocity, 'life': lifespan, 'max_life': lifespan, 'color': color})

    def _update_and_draw_particles(self):
        temp_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = 255 * (p['life'] / p['max_life'])
                color = (*p['color'][:3], int(alpha))
                radius = int(3 * (p['life'] / p['max_life']))
                if radius > 0:
                    pygame.gfxdraw.filled_circle(temp_surface, int(p['pos'][0]), int(p['pos'][1]), radius, color)
        
        self.screen.blit(temp_surface, (0, 0))

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert obs.dtype == np.uint8
        assert isinstance(info, dict)
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    env.validate_implementation()
    
    # --- To play manually ---
    # This requires a window, so we'll re-init pygame for display
    pygame.display.init()
    pygame.display.set_caption("Robot Repair")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()

    obs, info = env.reset()
    terminated = False
    
    print(env.user_guide)
    
    while True:
        # Action mapping for manual play
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        # The third action component is unused in the current logic, so we can set it to 0
        shift_held = 0 
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Progress: {info['repair_progress_percent']:.0f}%")
            # Wait for 'R' to reset or QUIT
            while True:
                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    break
        
        clock.tick(env.FPS)