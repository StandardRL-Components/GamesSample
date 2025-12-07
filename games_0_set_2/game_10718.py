import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:56:13.976528
# Source Brief: brief_00718.md
# Brief Index: 718
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a trio of drones
    to destroy enemy generators in a procedurally generated maze.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control a trio of drones to navigate a maze and destroy all enemy generators before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the drone trio. Press space to fire lasers."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_WALL = (100, 100, 120)
    COLOR_GENERATOR = (255, 50, 50)
    COLOR_GENERATOR_CORE = (255, 150, 150)
    COLOR_DRONE_1 = (50, 150, 255)
    COLOR_DRONE_2 = (50, 255, 150)
    COLOR_DRONE_3 = (255, 255, 100)
    COLOR_LASER = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    
    # Maze Generation
    MAZE_COLS = 32
    MAZE_ROWS = 20
    CELL_WIDTH = SCREEN_WIDTH // MAZE_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // MAZE_ROWS

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Game State Variables
        self.level = 0
        self.base_drone_speed = 4.0
        self._initialize_state()
        
        # Validate implementation
        # self.validate_implementation() # Optional validation call

    def _initialize_state(self):
        """Initializes or resets all game state variables."""
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 60.0
        
        self.drones = []
        self.generators = []
        self.maze_walls = []
        self.lasers = []
        self.particles = []
        
        self.prev_space_held = False

        # Level progression
        self.level += 1
        self.drone_speed = self.base_drone_speed + (self.level - 1) * 0.1
        self.num_generators = 10 + ((self.level - 1) // 2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.game_over: # Only increment level logic if a game was actually played
            # Persist drone speed on win, reset on loss
            if self._check_win_condition():
                self.base_drone_speed = self.drone_speed
            else:
                self.level = 0 # Reset level progress
                self.base_drone_speed = 4.0
        
        self._initialize_state()
        
        # Generate maze and place objects
        self.maze_walls, empty_cells = self._generate_maze()
        self._place_game_objects(empty_cells)
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0, True, False, info

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Calculate pre-move distance for reward
        pre_move_dist = self._get_total_drone_to_generator_dist()
        
        # Handle input and update drone state
        self._handle_input(movement, space_held)
        
        # Update game world
        self._update_game_state()

        # Calculate rewards
        post_move_dist = self._get_total_drone_to_generator_dist()
        dist_change_reward = (pre_move_dist - post_move_dist) * 0.01
        
        # Event rewards are handled inside _handle_input when generators are hit
        reward = self.score + dist_change_reward
        self.score = 0 # Reset score accumulator for next step

        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self._check_win_condition():
                reward += 100
            else:
                reward -= 100
        
        obs = self._get_observation()
        info = self._get_info()
        
        return (
            obs,
            reward,
            terminated,
            False, # Truncated is not used in this environment
            info
        )

    def _handle_input(self, movement, space_held):
        # --- Drone Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1

        if move_vec.length() > 0:
            for drone in self.drones:
                drone['dir'] = move_vec.copy()

        for drone in self.drones:
            new_pos = drone['pos'] + move_vec * self.drone_speed
            drone_rect = pygame.Rect(new_pos.x - 5, new_pos.y - 5, 10, 10)
            
            # Wall collision
            if not any(wall.colliderect(drone_rect) for wall in self.maze_walls):
                # Boundary collision
                if 5 <= new_pos.x <= self.SCREEN_WIDTH - 5 and 5 <= new_pos.y <= self.SCREEN_HEIGHT - 5:
                    drone['pos'] = new_pos

        # --- Firing ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            # sfx: drone_fire_command.wav
            for drone in self.drones:
                if self.np_random.random() < 0.5:
                    self._fire_laser(drone)
        self.prev_space_held = space_held
        
    def _fire_laser(self, drone):
        # sfx: laser_shoot.wav
        start_pos = drone['pos'].copy()
        direction = drone['dir'].copy()
        
        end_pos = start_pos + direction * 2000 # Effectively infinite length
        
        closest_hit_dist = float('inf')
        final_end_pos = end_pos

        # Check for hits against walls and generators
        entities_to_check = self.maze_walls + [g['rect'] for g in self.generators if g['alive']]
        for rect in entities_to_check:
            # Use pygame's line clipping which is robust
            clipped_line = rect.clipline(start_pos, end_pos)
            if clipped_line:
                hit_start, _ = clipped_line
                dist = start_pos.distance_to(hit_start)
                if dist < closest_hit_dist:
                    closest_hit_dist = dist
                    final_end_pos = pygame.Vector2(hit_start)

        # Check if the hit was a generator
        for gen in self.generators:
            if gen['alive'] and gen['rect'].collidepoint(final_end_pos):
                gen['alive'] = False
                self.score += 1.0 # Event reward
                # sfx: generator_explosion.wav
                self._create_explosion(gen['pos'], self.COLOR_GENERATOR, 50)
                break

        self.lasers.append({'start': start_pos, 'end': final_end_pos, 'life': 3})

    def _update_game_state(self):
        # Update timer
        self.time_remaining -= 1 / self.FPS
        
        # Update lasers
        self.lasers = [l for l in self.lasers if l['life'] > 0]
        for l in self.lasers:
            l['life'] -= 1
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Drag

    def _check_termination(self):
        if self.time_remaining <= 0 or self._check_win_condition():
            self.game_over = True
            return True
        if self.steps >= 1800: # Max episode length
            self.game_over = True
            return True
        return False

    def _check_win_condition(self):
        return not any(g['alive'] for g in self.generators)

    def _get_total_drone_to_generator_dist(self):
        alive_generators = [g['pos'] for g in self.generators if g['alive']]
        if not alive_generators:
            return 0
        
        total_dist = 0
        for drone in self.drones:
            min_dist = min(abs(drone['pos'].x - g.x) + abs(drone['pos'].y - g.y) for g in alive_generators)
            total_dist += min_dist
        return total_dist

    # --- MAZE AND OBJECT PLACEMENT ---
    def _generate_maze(self):
        w, h = self.MAZE_COLS, self.MAZE_ROWS
        visited = np.zeros((w, h), dtype=bool)
        walls = set()
        for x in range(w):
            for y in range(h):
                walls.add(((x, y), (x + 1, y)))
                walls.add(((x, y), (x, y + 1)))

        def walk(x, y):
            visited[x, y] = True
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            self.np_random.shuffle(neighbors)
            for nx, ny in neighbors:
                if 0 <= nx < w and 0 <= ny < h and not visited[nx, ny]:
                    wall = tuple(sorted(((x, y), (nx, ny))))
                    if wall in walls:
                        walls.remove(wall)
                    walk(nx, ny)

        walk(self.np_random.integers(w), self.np_random.integers(h))

        wall_rects = []
        for (x1, y1), (x2, y2) in walls:
            if x1 == x2: # Vertical wall
                rect = pygame.Rect(x1 * self.CELL_WIDTH - 1, y1 * self.CELL_HEIGHT, 2, self.CELL_HEIGHT)
            else: # Horizontal wall
                rect = pygame.Rect(x1 * self.CELL_WIDTH, y1 * self.CELL_HEIGHT - 1, self.CELL_WIDTH, 2)
            wall_rects.append(rect)
        
        empty_cells = [(x, y) for x in range(w) for y in range(h)]
        return wall_rects, empty_cells

    def _place_game_objects(self, empty_cells):
        self.np_random.shuffle(empty_cells)
        
        # Place drones
        drone_colors = [self.COLOR_DRONE_1, self.COLOR_DRONE_2, self.COLOR_DRONE_3]
        drone_spawn_cell = empty_cells.pop()
        center_pos = pygame.Vector2(
            drone_spawn_cell[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2,
            drone_spawn_cell[1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        )
        for i in range(3):
            angle = 2 * math.pi * i / 3
            offset = pygame.Vector2(math.cos(angle), math.sin(angle)) * 15
            self.drones.append({
                'pos': center_pos + offset,
                'dir': pygame.Vector2(1, 0),
                'color': drone_colors[i]
            })

        # Place generators
        for _ in range(self.num_generators):
            if not empty_cells: break
            gen_cell = empty_cells.pop(0)
            pos = pygame.Vector2(
                gen_cell[0] * self.CELL_WIDTH + self.CELL_WIDTH / 2,
                gen_cell[1] * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
            )
            self.generators.append({
                'pos': pos,
                'rect': pygame.Rect(pos.x - 8, pos.y - 8, 16, 16),
                'alive': True
            })
    
    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            p_color = (
                max(0, min(255, color[0] + self.np_random.integers(-20, 20))),
                max(0, min(255, color[1] + self.np_random.integers(-20, 20))),
                max(0, min(255, color[2] + self.np_random.integers(-20, 20)))
            )
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'color': p_color
            })

    # --- RENDERING ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_maze()
        self._render_generators()
        self._render_lasers()
        self._render_particles()
        self._render_drones()

    def _render_glow(self, pos, radius, color, alpha_factor=0.1):
        for i in range(radius, 0, -2):
            alpha = int(255 * (1 - i / radius) * alpha_factor)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), i, (*color, alpha))

    def _render_maze(self):
        for wall in self.maze_walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

    def _render_generators(self):
        for gen in self.generators:
            if gen['alive']:
                self._render_glow(gen['pos'], 20, self.COLOR_GENERATOR)
                pygame.gfxdraw.filled_circle(self.screen, int(gen['pos'].x), int(gen['pos'].y), 8, self.COLOR_GENERATOR)
                pygame.gfxdraw.filled_circle(self.screen, int(gen['pos'].x), int(gen['pos'].y), 4, self.COLOR_GENERATOR_CORE)
            else: # Render debris
                pygame.gfxdraw.filled_circle(self.screen, int(gen['pos'].x), int(gen['pos'].y), 6, (60, 60, 70))


    def _render_drones(self):
        for drone in self.drones:
            self._render_glow(drone['pos'], 15, drone['color'])
            p1 = drone['pos'] + drone['dir'] * 8
            p2 = drone['pos'] + drone['dir'].rotate(-120) * 6
            p3 = drone['pos'] + drone['dir'].rotate(120) * 6
            points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
            pygame.gfxdraw.aapolygon(self.screen, points, drone['color'])
            pygame.gfxdraw.filled_polygon(self.screen, points, drone['color'])

    def _render_lasers(self):
        for l in self.lasers:
            alpha = int(255 * (l['life'] / 3.0))
            # color = (*self.COLOR_LASER, alpha) # This color is not used
            pygame.draw.line(self.screen, self.COLOR_LASER, l['start'], l['end'], 2)

    def _render_particles(self):
        for p in self.particles:
            # alpha = int(255 * (p['life'] / 30.0)) # Alpha is not used here
            # color = (*p['color'], alpha)
            size = int(max(1, 5 * (p['life'] / 30.0)))
            pygame.draw.rect(self.screen, p['color'], (p['pos'].x, p['pos'].y, size, size))

    def _render_ui(self):
        # Generators remaining
        gen_text = f"GENERATORS: {sum(1 for g in self.generators if g['alive'])}"
        text_surf = self.font_small.render(gen_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Time remaining
        time_text = f"TIME: {max(0, int(self.time_remaining))}"
        text_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))
        
        # Level
        level_text = f"LEVEL: {self.level}"
        text_surf = self.font_small.render(level_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, self.SCREEN_HEIGHT - text_surf.get_height() - 10))
        
        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "LEVEL COMPLETE" if self._check_win_condition() else "TIME UP"
            msg_surf = self.font_large.render(msg, True, self.COLOR_TEXT)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": sum(1 for g in self.generators if not g['alive']),
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "level": self.level,
            "drones_alive": len(self.drones)
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
        self.level = 0 # Reset level to 1 for a clean test
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

if __name__ == '__main__':
    # Example usage:
    # Un-comment the line below to run with a display window
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Manual play loop
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Drone Maze")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # no-op, released, released
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Final Info: {info}")
            # Keep showing the final frame until R is pressed
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()