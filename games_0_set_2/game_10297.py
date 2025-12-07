import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:09:40.024277
# Source Brief: brief_00297.md
# Brief Index: 297
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, life, size, angle, speed):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        # Fade out effect
        alpha = max(0, int(255 * (self.life / self.max_life)))
        self.current_color = (*self.color, alpha)

    def draw(self, surface):
        if self.life > 0:
            pygame.draw.circle(surface, self.current_color, (int(self.x), int(self.y)), int(self.size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    game_description = (
        "Navigate a procedurally generated maze as a size-changing cube. "
        "Collect crystals to alter your size and speed, and reach the exit before time runs out."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move your cube through the maze. Reach the green exit before the timer runs out."
    )
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = 60 * self.FPS # 60 seconds

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (70, 80, 100)
        self.COLOR_WALL_TOP = (90, 100, 120)
        self.COLOR_PLAYER = (255, 60, 60)
        self.COLOR_PLAYER_GLOW = (255, 100, 100)
        self.COLOR_EXIT = (60, 255, 150)
        self.COLOR_EXIT_GLOW = (150, 255, 200)
        self.COLOR_SMALL_CRYSTAL = (100, 200, 255)
        self.COLOR_LARGE_CRYSTAL = (200, 100, 255)
        self.COLOR_TEXT = (230, 230, 240)

        # Maze settings
        self.MAZE_COLS, self.MAZE_ROWS = 16, 10
        self.CELL_SIZE = 40

        # Player settings
        self.PLAYER_BASE_SIZE = self.CELL_SIZE * 0.4
        self.PLAYER_BASE_SPEED = 3.0
        self.MIN_SIZE_MULTIPLIER = 0.25
        self.MAX_SIZE_MULTIPLIER = 2.5

        # Item settings
        self.NUM_SMALL_CRYSTALS = 10
        self.NUM_LARGE_CRYSTALS = 5

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_size_multiplier = 1.0
        self.walls = []
        self.small_crystals = []
        self.large_crystals = []
        self.exit_rect = None
        self.particles = []
        self.previous_distance_to_exit = 0.0
        self.render_mode = render_mode
        self.human_screen = None

        if self.render_mode == "human":
            self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_size_multiplier = 1.0
        self.particles.clear()

        self._generate_maze()
        self._place_items_and_player()

        self.previous_distance_to_exit = self._get_distance_to_exit()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held and shift_held are ignored as per the brief.

        # --- Game Logic ---
        self.steps += 1
        
        # 1. Handle player movement
        self._move_player(movement)
        
        # 2. Handle collisions and collect rewards
        collision_reward = self._handle_collisions()

        # 3. Calculate distance-based reward
        current_distance = self._get_distance_to_exit()
        distance_reward = 0.0
        if current_distance < self.previous_distance_to_exit:
            distance_reward = 0.01
        elif current_distance > self.previous_distance_to_exit:
            distance_reward = -0.01
        self.previous_distance_to_exit = current_distance

        # 4. Update particles
        self._update_particles()
        
        # 5. Check for termination
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        # 6. Calculate final reward
        reward = distance_reward + collision_reward
        if terminated:
            if self.exit_rect.collidepoint(self.player_pos):
                reward += 100 # Reached exit
        if truncated:
             reward -= 10 # Time ran out
        
        self.score += reward

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _generate_maze(self):
        w, h = self.MAZE_COLS, self.MAZE_ROWS
        visited = np.zeros((h, w), dtype=bool)
        # grid stores wall state: 1=right, 2=bottom, 4=left, 8=top
        grid = np.zeros((h, w), dtype=int)
        
        def carve(x, y):
            visited[y, x] = True
            directions = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            self.np_random.shuffle(directions)
            for nx, ny in directions:
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if nx == x - 1: grid[y, x] |= 4; grid[ny, nx] |= 1
                    if nx == x + 1: grid[y, x] |= 1; grid[ny, nx] |= 4
                    if ny == y - 1: grid[y, x] |= 8; grid[ny, ny] |= 2
                    if ny == y + 1: grid[y, x] |= 2; grid[ny, ny] |= 8
                    carve(nx, ny)
        
        carve(self.np_random.integers(w), self.np_random.integers(h))
        self.maze_grid = grid
        
        self.walls.clear()
        # Inner walls
        for y in range(h):
            for x in range(w):
                if not (grid[y, x] & 1): # No right passage
                    self.walls.append(pygame.Rect((x + 1) * self.CELL_SIZE - 1, y * self.CELL_SIZE, 2, self.CELL_SIZE))
                if not (grid[y, x] & 2): # No bottom passage
                    self.walls.append(pygame.Rect(x * self.CELL_SIZE, (y + 1) * self.CELL_SIZE - 1, self.CELL_SIZE, 2))
        
        # Add outer boundary walls
        self.walls.append(pygame.Rect(0, 0, self.WIDTH, 2))
        self.walls.append(pygame.Rect(0, 0, 2, self.HEIGHT))
        self.walls.append(pygame.Rect(self.WIDTH-2, 0, 2, self.HEIGHT))
        self.walls.append(pygame.Rect(0, self.HEIGHT-2, self.WIDTH, 2))

    def _place_items_and_player(self):
        open_cells = []
        for y in range(self.MAZE_ROWS):
            for x in range(self.MAZE_COLS):
                open_cells.append((x, y))
        
        self.np_random.shuffle(open_cells)
        
        # Player start
        px, py = open_cells.pop()
        self.player_pos = np.array([px * self.CELL_SIZE + self.CELL_SIZE / 2, py * self.CELL_SIZE + self.CELL_SIZE / 2])
        
        # Exit
        ex, ey = open_cells.pop()
        self.exit_rect = pygame.Rect(ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)

        # Crystals
        self.small_crystals.clear()
        self.large_crystals.clear()
        for _ in range(self.NUM_SMALL_CRYSTALS):
            if not open_cells: break
            cx, cy = open_cells.pop()
            self.small_crystals.append(pygame.Rect(cx * self.CELL_SIZE + self.CELL_SIZE/2 - 5, cy * self.CELL_SIZE + self.CELL_SIZE/2 - 5, 10, 10))
        for _ in range(self.NUM_LARGE_CRYSTALS):
            if not open_cells: break
            cx, cy = open_cells.pop()
            self.large_crystals.append(pygame.Rect(cx * self.CELL_SIZE + self.CELL_SIZE/2 - 8, cy * self.CELL_SIZE + self.CELL_SIZE/2 - 8, 16, 16))

    def _move_player(self, movement):
        speed = self.PLAYER_BASE_SPEED / max(0.5, self.player_size_multiplier)
        move_vec = np.array([0.0, 0.0])
        if movement == 1: move_vec[1] = -speed # Up
        elif movement == 2: move_vec[1] = speed  # Down
        elif movement == 3: move_vec[0] = -speed # Left
        elif movement == 4: move_vec[0] = speed  # Right
        
        new_pos = self.player_pos + move_vec
        
        # Collision detection with walls
        player_size = self.PLAYER_BASE_SIZE * self.player_size_multiplier
        new_player_rect = pygame.Rect(new_pos[0] - player_size / 2, new_pos[1] - player_size / 2, player_size, player_size)

        collided = False
        for wall in self.walls:
            if new_player_rect.colliderect(wall):
                collided = True
                # Simple slide response
                old_player_rect = pygame.Rect(self.player_pos[0] - player_size / 2, self.player_pos[1] - player_size / 2, player_size, player_size)
                
                # Try moving only on X axis
                temp_rect_x = old_player_rect.copy()
                temp_rect_x.x += move_vec[0]
                if not temp_rect_x.colliderect(wall):
                    self.player_pos[0] += move_vec[0]
                
                # Try moving only on Y axis
                temp_rect_y = old_player_rect.copy()
                temp_rect_y.y += move_vec[1]
                if not temp_rect_y.colliderect(wall):
                    self.player_pos[1] += move_vec[1]

                break
        
        if not collided:
            self.player_pos = new_pos
        
        # Clamp to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], player_size/2, self.WIDTH - player_size/2)
        self.player_pos[1] = np.clip(self.player_pos[1], player_size/2, self.HEIGHT - player_size/2)
        
    def _handle_collisions(self):
        reward = 0
        player_size = self.PLAYER_BASE_SIZE * self.player_size_multiplier
        player_rect = pygame.Rect(self.player_pos[0] - player_size / 2, self.player_pos[1] - player_size / 2, player_size, player_size)

        # Wall collisions (for penalty, actual response is in _move_player)
        for wall in self.walls:
            if player_rect.colliderect(wall):
                self.player_size_multiplier = max(self.MIN_SIZE_MULTIPLIER, self.player_size_multiplier * 0.995) # Shrink 0.5%
                reward -= 0.05
                self._create_particles(player_rect.center, (200,200,200), 2, 1, 10) # Wall hit sparks
                break

        # Crystal collisions
        for crystal in self.small_crystals[:]:
            if player_rect.colliderect(crystal):
                self.small_crystals.remove(crystal)
                self.player_size_multiplier = max(self.MIN_SIZE_MULTIPLIER, self.player_size_multiplier * 0.8) # Shrink 20%
                reward += 5.0
                self._create_particles(crystal.center, self.COLOR_SMALL_CRYSTAL, 20, 3, 30)
        
        for crystal in self.large_crystals[:]:
            if player_rect.colliderect(crystal):
                self.large_crystals.remove(crystal)
                self.player_size_multiplier = min(self.MAX_SIZE_MULTIPLIER, self.player_size_multiplier * 1.4) # Grow 40%
                reward += 5.0
                self._create_particles(crystal.center, self.COLOR_LARGE_CRYSTAL, 30, 4, 40)
                
        return reward

    def _create_particles(self, pos, color, count, speed, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            p_speed = self.np_random.uniform(0.5, 1.0) * speed
            p_life = self.np_random.uniform(0.8, 1.2) * life
            p_size = self.np_random.uniform(1, 4)
            self.particles.append(Particle(pos[0], pos[1], color, p_life, p_size, angle, p_speed))
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def _check_termination(self):
        player_size = self.PLAYER_BASE_SIZE * self.player_size_multiplier
        player_rect = pygame.Rect(self.player_pos[0] - player_size / 2, self.player_pos[1] - player_size / 2, player_size, player_size)
        
        if self.exit_rect.colliderect(player_rect):
            self.game_over = True
            return True
        return False

    def render(self):
        return self._get_observation()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        if self.render_mode == "human":
            if self.human_screen is None:
                self.human_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.human_screen.blit(self.screen, (0,0))
            pygame.display.flip()
            self.clock.tick(self.FPS)

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render Exit
        glow_size = int(self.CELL_SIZE * 1.5 + abs(math.sin(self.steps / 20)) * 5)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.COLOR_EXIT_GLOW, 50), (glow_size//2, glow_size//2), glow_size//2)
        self.screen.blit(glow_surf, (self.exit_rect.centerx - glow_size//2, self.exit_rect.centery - glow_size//2))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect.inflate(-10, -10))

        # Render Walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)
            # 3D-ish effect
            if wall.width > wall.height: # Horizontal wall
                 pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, (wall.x, wall.y, wall.width, wall.height * 0.4))
            else: # Vertical wall
                 pygame.draw.rect(self.screen, self.COLOR_WALL_TOP, (wall.x, wall.y, wall.width * 0.4, wall.height))


        # Render Crystals
        for crystal in self.small_crystals:
            pygame.gfxdraw.filled_circle(self.screen, crystal.centerx, crystal.centery, crystal.width // 2, self.COLOR_SMALL_CRYSTAL)
            pygame.gfxdraw.aacircle(self.screen, crystal.centerx, crystal.centery, crystal.width // 2, self.COLOR_SMALL_CRYSTAL)
            # Highlight
            pygame.gfxdraw.filled_circle(self.screen, crystal.centerx-1, crystal.centery-1, crystal.width // 4, (200, 240, 255))
            
        for crystal in self.large_crystals:
            c = crystal.center
            s = crystal.width
            points = [(c[0], c[1]-s/2), (c[0]+s/2, c[1]), (c[0], c[1]+s/2), (c[0]-s/2, c[1])]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_LARGE_CRYSTAL)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_LARGE_CRYSTAL)
            # Highlight
            pygame.gfxdraw.filled_polygon(self.screen, [(c[0]-s*0.1, c[1]-s*0.4), (c[0]+s*0.4, c[1]-s*0.1), (c[0], c[1])], (240, 180, 255))

        # Render Particles
        for p in self.particles:
            p.draw(self.screen)

        # Render Player
        size = self.PLAYER_BASE_SIZE * self.player_size_multiplier
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Glow
        glow_size = int(size * 1.8)
        glow_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        alpha = 100 + int(abs(math.sin(self.steps / 10)) * 40)
        pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, alpha), (glow_size//2, glow_size//2), glow_size//2)
        self.screen.blit(glow_surf, (pos[0] - glow_size//2, pos[1] - glow_size//2))
        
        # 3D Cube effect
        player_rect = pygame.Rect(pos[0] - size/2, pos[1] - size/2, size, size)
        offset = int(size * 0.15)
        darker_color = (max(0, self.COLOR_PLAYER[0]-50), max(0, self.COLOR_PLAYER[1]-50), max(0, self.COLOR_PLAYER[2]-50))
        pygame.draw.rect(self.screen, darker_color, player_rect.move(offset, offset))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
    def _render_ui(self):
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        seconds = time_left // self.FPS
        text = f"TIME: {seconds:02d}"
        text_surf = self.font.render(text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 15, 10))
        
        # Score
        score_text = f"SCORE: {self.score:.2f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (15, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_size": self.player_size_multiplier,
            "distance_to_exit": self.previous_distance_to_exit,
        }

    def _get_distance_to_exit(self):
        return np.linalg.norm(self.player_pos - np.array(self.exit_rect.center))

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # Example usage:
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    
    # Manual play loop
    # 0=none, 1=up, 2=down, 3=left, 4=right
    # 0=released, 1=held
    action = [0, 0, 0] 
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        action[0] = 0 # Default to no-op
        if keys[pygame.K_UP] or keys[pygame.K_w]: action[0] = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: action[0] = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: action[0] = 4
        
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f} in {info['steps']} steps.")
            obs, info = env.reset()
            done = True # End after one game

    env.close()