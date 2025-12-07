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
    metadata = {"render_modes": ["rgb_array", "human"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to place a crystal. Shift to cycle crystal type (Blue/Green/Purple)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Place reflective and splitting crystals to guide a laser beam to the target before time runs out or it bounces too many times."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = WIDTH // GRID_COLS
    MAX_STEPS = 300 # A reasonable number of actions for a puzzle
    MAX_BOUNCES = 10

    # --- Colors ---
    COLOR_BG = (26, 26, 46)
    COLOR_WALL = (74, 74, 94)
    COLOR_GRID = (40, 40, 60)
    COLOR_TARGET = (255, 215, 0)
    COLOR_TARGET_GLOW = (255, 215, 0, 50)
    COLOR_LASER = (255, 0, 127)
    COLOR_LASER_GLOW = (255, 0, 127, 100)
    COLOR_UI_TEXT = (255, 255, 255)
    CRYSTAL_COLORS = [(0, 255, 255), (0, 255, 127), (191, 0, 255)] # Blue, Green, Purple
    CRYSTAL_GLOWS = [(0, 255, 255, 50), (0, 255, 127, 50), (191, 0, 255, 50)]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Etc...
        self.render_mode = render_mode
        
        # Initialize state variables
        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.last_reward = 0

        self.cursor_pos = np.array([self.GRID_COLS // 2, self.GRID_ROWS // 2])
        self.selected_crystal = 0  # 0: Blue, 1: Green, 2: Purple
        self.crystals = [] # List of (pos, type) tuples

        self.laser_source = np.array([self.CELL_SIZE / 2, self.HEIGHT / 2], dtype=float)
        self.target_pos = np.array([self.WIDTH - self.CELL_SIZE * 1.5, self.HEIGHT / 2], dtype=float)
        
        self.hit_target = False
        self.bounce_count = 0
        self.laser_paths = []
        self.particles = []

        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._calculate_laser_path()
        self.dist_to_target_before_action = self._get_laser_endpoint_dist_to_target()

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean

        # Update game logic
        self.steps += 1
        self.time_remaining -= 1
        reward = -0.01  # Cost per step to encourage efficiency

        # --- Handle Actions ---
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Cycle crystal on press (detects rising edge)
        if shift_held and not self.prev_shift_held:
            self.selected_crystal = (self.selected_crystal + 1) % 3
            # sfx: crystal_cycle.wav

        # Place crystal on press (detects rising edge)
        if space_held and not self.prev_space_held:
            # sfx: crystal_place.wav
            is_valid_placement = True
            cursor_world_pos = self.cursor_pos * self.CELL_SIZE + self.CELL_SIZE / 2
            if np.linalg.norm(cursor_world_pos - self.target_pos) < self.CELL_SIZE:
                is_valid_placement = False
            if np.linalg.norm(cursor_world_pos - self.laser_source) < self.CELL_SIZE:
                 is_valid_placement = False
            for c_pos, _ in self.crystals:
                if np.array_equal(self.cursor_pos, c_pos):
                    is_valid_placement = False
                    break
            
            if is_valid_placement:
                self.crystals.append((self.cursor_pos.copy(), self.selected_crystal))
                self._calculate_laser_path()
                
                new_dist = self._get_laser_endpoint_dist_to_target()
                if new_dist < self.dist_to_target_before_action:
                    reward += 1.0 # Closer to target
                else:
                    reward -= 1.0 # Further from target
                self.dist_to_target_before_action = new_dist

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        self._update_particles()
        terminated = self._check_termination()

        if self.hit_target:
            reward += 100
            # sfx: win.wav
        elif terminated and self.time_remaining > 0: # Failed due to bounces
            reward -= 100
            # sfx: lose.wav
        elif terminated and self.time_remaining <= 0: # Failed due to time
            reward -= 100

        self.score += reward
        self.last_reward = reward

        # MUST return exactly this 5-tuple
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_laser_path(self):
        self.laser_paths = []
        self.bounce_count = 0
        self.hit_target = False

        q = [(self.laser_source, np.array([1.0, 0.0]))] # Queue of (origin, direction)
        
        processed_rays = 0
        while q and processed_rays < 50: # Safety break for complex splits
            processed_rays += 1
            origin, direction = q.pop(0)
            
            current_origin = origin
            for _ in range(self.MAX_BOUNCES + 2):
                if self.bounce_count > self.MAX_BOUNCES:
                    break
                
                intersections = []
                
                # Walls
                if direction[0] != 0:
                    t = (0 - current_origin[0]) / direction[0]
                    if t > 1e-6: intersections.append((t, 'wall_left'))
                    t = (self.WIDTH - current_origin[0]) / direction[0]
                    if t > 1e-6: intersections.append((t, 'wall_right'))
                if direction[1] != 0:
                    t = (0 - current_origin[1]) / direction[1]
                    if t > 1e-6: intersections.append((t, 'wall_top'))
                    t = (self.HEIGHT - current_origin[1]) / direction[1]
                    if t > 1e-6: intersections.append((t, 'wall_bottom'))
                
                # Crystals (as spheres for intersection)
                for c_grid_pos, c_type in self.crystals:
                    c_pos = c_grid_pos * self.CELL_SIZE + self.CELL_SIZE / 2
                    oc = current_origin - c_pos
                    a = np.dot(direction, direction)
                    b = 2 * np.dot(oc, direction)
                    c = np.dot(oc, oc) - (self.CELL_SIZE/2)**2
                    discriminant = b**2 - 4*a*c
                    if discriminant >= 0:
                        t = (-b - math.sqrt(discriminant)) / (2*a)
                        if t > 1e-6:
                            intersections.append((t, ('crystal', c_pos, c_type)))

                # Target
                oc = current_origin - self.target_pos
                a = np.dot(direction, direction)
                b = 2 * np.dot(oc, direction)
                c = np.dot(oc, oc) - (self.CELL_SIZE/2)**2
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    t = (-b - math.sqrt(discriminant)) / (2*a)
                    if t > 1e-6:
                        intersections.append((t, 'target'))

                if not intersections: break
                
                intersections.sort(key=lambda x: x[0])
                t_min, obj = intersections[0]
                
                end_point = current_origin + t_min * direction
                self.laser_paths.append((current_origin, end_point))
                self.bounce_count += 1

                if obj == 'target':
                    self.hit_target = True
                    break
                elif isinstance(obj, str) and 'wall' in obj:
                    if 'left' in obj or 'right' in obj:
                        direction[0] *= -1
                    if 'top' in obj or 'bottom' in obj:
                        direction[1] *= -1
                    current_origin = end_point
                    # sfx: bounce_wall.wav
                elif isinstance(obj, tuple) and obj[0] == 'crystal':
                    _, _, c_type = obj
                    # sfx: bounce_crystal.wav
                    if c_type == 0:  # Blue: Reflects across y=-x line
                        direction = np.array([-direction[1], -direction[0]])
                    elif c_type == 1:  # Green: Rotates 90 degrees clockwise
                        direction = np.array([direction[1], -direction[0]])
                    elif c_type == 2:  # Purple: Splits, one straight, one 90 deg
                        q.append((end_point, np.array([direction[1], -direction[0]])))
                    current_origin = end_point
            
            if self.hit_target: break
        
        self.bounce_count = min(self.bounce_count, self.MAX_BOUNCES + 1)

    def _get_laser_endpoint_dist_to_target(self):
        if not self.laser_paths:
            return np.linalg.norm(self.laser_source - self.target_pos)
        
        last_points = {}
        for start, end in self.laser_paths:
            last_points[tuple(start)] = end
        
        endpoints = set(tuple(p) for p in last_points.values())
        min_dist = float('inf')
        for end_point in endpoints:
            dist = np.linalg.norm(np.array(end_point) - self.target_pos)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _check_termination(self):
        terminated = self.hit_target or self.bounce_count > self.MAX_BOUNCES or self.time_remaining <= 0
        if terminated:
            self.game_over = True
        return terminated

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
        # Draw grid
        for i in range(1, self.GRID_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.HEIGHT))
        for i in range(1, self.GRID_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i * self.CELL_SIZE), (self.WIDTH, i * self.CELL_SIZE))

        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw target
        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        pygame.gfxdraw.filled_circle(self.screen, tx, ty, self.CELL_SIZE // 2 + 5, self.COLOR_TARGET_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, tx, ty, self.CELL_SIZE // 2, self.COLOR_TARGET)
        pygame.gfxdraw.aacircle(self.screen, tx, ty, self.CELL_SIZE // 2, self.COLOR_TARGET)

        # Draw laser source
        sx, sy = int(self.laser_source[0]), int(self.laser_source[1])
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, 5, self.COLOR_LASER)
        
        # Draw particles
        for p in self.particles:
            pos, life, max_life = p
            alpha = int(255 * (life / max_life))
            color = (*self.COLOR_LASER, alpha)
            size = int(3 * (life / max_life))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), size, color)

        # Draw laser paths
        for start, end in self.laser_paths:
            x1, y1 = int(start[0]), int(start[1])
            x2, y2 = int(end[0]), int(end[1])
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, (x1, y1), (x2, y2), 5)
            pygame.draw.line(self.screen, self.COLOR_LASER, (x1, y1), (x2, y2), 2)
            
            if self.np_random.random() < 0.5:
                dist = np.linalg.norm(end-start)
                num_particles = int(dist / 20)
                for i in range(num_particles):
                    life = self.np_random.integers(10, 20)
                    pos = start + (end-start) * (i / max(1, num_particles))
                    self.particles.append([pos, life, life])

        # Draw crystals
        for c_pos, c_type in self.crystals:
            center_x = int(c_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(c_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2)
            size = self.CELL_SIZE // 3
            color = self.CRYSTAL_COLORS[c_type]
            glow = self.CRYSTAL_GLOWS[c_type]
            
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, size + 4, glow)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, size, color)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, size, color)
        
        # Draw cursor
        if not self.game_over:
            cursor_x = int(self.cursor_pos[0] * self.CELL_SIZE)
            cursor_y = int(self.cursor_pos[1] * self.CELL_SIZE)
            cursor_color = self.CRYSTAL_COLORS[self.selected_crystal]
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((*cursor_color, 50))
            self.screen.blit(s, (cursor_x, cursor_y))
            pygame.draw.rect(self.screen, cursor_color, (cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE), 2)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[1] > 0]
        for p in self.particles:
            p[1] -= 1 # Decrement life

    def _render_ui(self):
        # Time remaining
        time_text = self.font_ui.render(f"TIME: {self.time_remaining}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Bounces
        bounce_color = self.COLOR_UI_TEXT if self.bounce_count <= self.MAX_BOUNCES else (255, 50, 50)
        bounce_text = self.font_ui.render(f"BOUNCES: {self.bounce_count}/{self.MAX_BOUNCES}", True, bounce_color)
        self.screen.blit(bounce_text, (10, 10))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

        # Game Over
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.hit_target:
                msg = "TARGET HIT!"
                color = self.COLOR_TARGET
            else:
                msg = "FAILED"
                color = (255, 50, 50)
            
            end_text = self.font_big.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "bounces": self.bounce_count,
            "crystals_placed": len(self.crystals),
            "hit_target": self.hit_target
        }

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block is for human play and is not run by the evaluation system.
    # It allows you to test the environment manually.
    
    # Un-comment the line below to run with a visible display
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    # --- Human Play Setup ---
    pygame.display.set_caption("Laser Cavern")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    game_over = False
    
    # Store previous key states to detect single presses
    prev_keys = pygame.key.get_pressed()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                game_over = False
                print("--- Game Reset ---")

        if not game_over:
            current_keys = pygame.key.get_pressed()
            
            # --- Map keys to actions ---
            # This is a bit different from the agent's perspective.
            # We check for key *presses* (rising edge) for placing/cycling
            # to make it more playable for a human.
            
            action = np.array([0, 0, 0]) # no-op
            
            # Continuous movement
            if current_keys[pygame.K_UP]: action[0] = 1
            elif current_keys[pygame.K_DOWN]: action[0] = 2
            elif current_keys[pygame.K_LEFT]: action[0] = 3
            elif current_keys[pygame.K_RIGHT]: action[0] = 4
            
            # Single press actions
            space_pressed = current_keys[pygame.K_SPACE] and not prev_keys[pygame.K_SPACE]
            shift_pressed = (current_keys[pygame.K_LSHIFT] or current_keys[pygame.K_RSHIFT]) and not (prev_keys[pygame.K_LSHIFT] or prev_keys[pygame.K_RSHIFT])

            # To match the MultiDiscrete space, the action is [move, place, cycle]
            # The environment's step function handles the rising-edge logic itself.
            # So we just pass the current state of the keys.
            action[1] = 1 if current_keys[pygame.K_SPACE] else 0
            action[2] = 1 if (current_keys[pygame.K_LSHIFT] or current_keys[pygame.K_RSHIFT]) else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            game_over = terminated

            if reward != 0:
                print(f"Step {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

            if game_over:
                print(f"Game Over! Press R to restart. Final Score: {info['score']:.2f}")
                reason = "Hit Target" if info['hit_target'] else ('Max Bounces' if info['bounces'] > GameEnv.MAX_BOUNCES else 'Time Up')
                print(f"Reason: {reason}")

            prev_keys = current_keys

        # --- Render the game screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(20) # Limit frame rate for human play

    env.close()