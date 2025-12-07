
# Generated: 2025-08-27T16:16:37.651466
# Source Brief: brief_01174.md
# Brief Index: 1174

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑ to jump."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Escape a procedurally generated haunted house by dodging ghosts and reaching the exit within a time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.TIME_PER_STAGE = 60 * self.FPS
        self.MAX_STAGES = 3

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_WALL = (40, 50, 80)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (255, 255, 255, 50)
        self.COLOR_EXIT = (255, 255, 0)
        self.COLOR_EXIT_GLOW = (255, 255, 0, 30)
        self.GHOST_COLORS = [
            (255, 80, 80),   # Red
            (80, 255, 80),   # Green
            (80, 80, 255)    # Blue
        ]
        self.COLOR_TEXT = (220, 220, 220)
        
        # Physics
        self.GRAVITY = 0.8
        self.PLAYER_SPEED = 4
        self.JUMP_STRENGTH = -12
        self.TILE_SIZE = 40
        self.PLAYER_WIDTH = 20
        self.PLAYER_HEIGHT = 20
        self.GHOST_SIZE = 24

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
        self.font_ui = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 36, bold=True)
        
        # State variables (initialized in reset)
        self.player = None
        self.player_vel = None
        self.on_ground = None
        self.ghosts = []
        self.ghost_trails = []
        self.walls = []
        self.exit_door = None
        self.particles = []
        
        self.stage = 0
        self.timer = 0
        self.total_score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.ghost_near_miss_flags = []
        
        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation()

    def _generate_level(self):
        grid_w, grid_h = self.WIDTH // self.TILE_SIZE, self.HEIGHT // self.TILE_SIZE
        grid = np.ones((grid_w, grid_h), dtype=int)

        # Carve a basic floor
        start_x, start_y = 1, grid_h - 2
        grid[1:grid_w-1, start_y] = 0

        # Create a winding path
        path_x, path_y = start_x, start_y
        for _ in range(grid_w * 2):
            path_x += self.np_random.choice([-1, 1, 1]) # Bias to move right
            path_x = np.clip(path_x, 1, grid_w - 2)
            grid[path_x, path_y] = 0
            if path_y > 1:
                grid[path_x, path_y - 1] = 0 # Headroom

        # Place the exit at the end of the path
        self.exit_door = pygame.Rect(
            path_x * self.TILE_SIZE,
            (path_y - 2) * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE * 2
        )

        # Add random floating platforms
        for _ in range(self.np_random.integers(4, 7)):
            plat_len = self.np_random.integers(2, 5)
            plat_x = self.np_random.integers(1, grid_w - plat_len - 1)
            plat_y = self.np_random.integers(2, grid_h - 4)
            grid[plat_x : plat_x + plat_len, plat_y] = 0

        # Convert grid to wall rects
        self.walls = []
        for x in range(grid_w):
            for y in range(grid_h):
                if grid[x, y] == 1:
                    self.walls.append(pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE))

        # Place player
        self.player = pygame.Rect(start_x * self.TILE_SIZE, (start_y - 1) * self.TILE_SIZE, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        self.player_vel = [0, 0]

    def _setup_stage(self):
        self.stage += 1
        self.timer = self.TIME_PER_STAGE
        self._generate_level()

        # Initialize ghosts
        self.ghosts = []
        self.ghost_trails = []
        self.ghost_near_miss_flags = [False] * 3
        base_speed = 1.0 + (self.stage - 1) * 0.2
        for i in range(3):
            patrol_start_x = self.np_random.integers(self.TILE_SIZE, self.WIDTH // 2)
            patrol_end_x = self.np_random.integers(self.WIDTH // 2, self.WIDTH - self.TILE_SIZE * 2)
            start_pos_y = self.np_random.integers(self.TILE_SIZE, self.HEIGHT - self.TILE_SIZE * 3)
            
            ghost = {
                "rect": pygame.Rect(patrol_start_x, start_pos_y, self.GHOST_SIZE, self.GHOST_SIZE),
                "speed": base_speed * self.np_random.uniform(0.8, 1.2),
                "patrol_start": patrol_start_x,
                "patrol_end": patrol_end_x,
                "direction": 1,
                "color": self.GHOST_COLORS[i]
            }
            self.ghosts.append(ghost)
            self.ghost_trails.append([])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.stage = 0
        self.total_score = 0
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.particles = []
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = 0.0

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.player_vel[0] = 0
        if movement == 3:  # Left
            self.player_vel[0] = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel[0] = self.PLAYER_SPEED
        
        if movement == 1 and self.on_ground: # Jump
            self.player_vel[1] = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump
            self._create_jump_particles()

        # --- Update Game Logic ---
        self._update_player()
        self._update_ghosts()
        self._update_particles()
        
        # --- Reward Calculation ---
        reward += 0.1  # Survival reward
        if movement == 0:
            reward -= 0.2 # Inactivity penalty

        # --- Termination and Collision Checks ---
        terminated = False
        
        # Ghost interactions
        danger_zone_scale = 4
        for i, ghost in enumerate(self.ghosts):
            danger_zone = ghost["rect"].inflate(self.GHOST_SIZE * danger_zone_scale, self.GHOST_SIZE * danger_zone_scale)
            
            if self.player.colliderect(danger_zone):
                if not self.ghost_near_miss_flags[i]:
                    reward += 1.0 # Near miss reward
                    self.ghost_near_miss_flags[i] = True
            else:
                self.ghost_near_miss_flags[i] = False
            
            if self.player.colliderect(ghost["rect"]):
                reward = -5.0
                self.game_over = True
                terminated = True
                # sfx: player_death
                break
        
        if terminated: # Don't process other terminations if already dead
            self.total_score += reward
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Exit collision
        if self.player.colliderect(self.exit_door):
            reward += 100.0
            # sfx: stage_clear
            if self.stage >= self.MAX_STAGES:
                self.game_won = True
                terminated = True
            else:
                self._setup_stage() # Progress to next stage
        
        # Timeout
        if self.timer <= 0:
            reward = -10.0
            self.game_over = True
            terminated = True
            # sfx: timeout

        self.total_score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_player(self):
        # Apply gravity
        self.player_vel[1] += self.GRAVITY
        self.player_vel[1] = min(self.player_vel[1], 10) # Terminal velocity

        # Move and collide horizontally
        self.player.x += int(self.player_vel[0])
        for wall in self.walls:
            if self.player.colliderect(wall):
                if self.player_vel[0] > 0: self.player.right = wall.left
                elif self.player_vel[0] < 0: self.player.left = wall.right
        
        # Move and collide vertically
        self.on_ground = False
        self.player.y += int(self.player_vel[1])
        for wall in self.walls:
            if self.player.colliderect(wall):
                if self.player_vel[1] > 0:
                    self.player.bottom = wall.top
                    self.on_ground = True
                    self.player_vel[1] = 0
                elif self.player_vel[1] < 0:
                    self.player.top = wall.bottom
                    self.player_vel[1] = 0
        
        # Screen bounds
        self.player.left = max(0, self.player.left)
        self.player.right = min(self.WIDTH, self.player.right)
        if self.player.top > self.HEIGHT: # Fell off screen
             self.game_over = True

    def _update_ghosts(self):
        for i, ghost in enumerate(self.ghosts):
            # Update trail
            trail = self.ghost_trails[i]
            trail.append(ghost["rect"].copy())
            if len(trail) > 10:
                trail.pop(0)

            # Move
            ghost["rect"].x += ghost["speed"] * ghost["direction"]
            
            # Patrol logic
            if ghost["rect"].right >= ghost["patrol_end"] or ghost["rect"].left <= ghost["patrol_start"]:
                ghost["direction"] *= -1

    def _create_jump_particles(self):
        for _ in range(8):
            self.particles.append({
                "pos": [self.player.centerx, self.player.bottom],
                "vel": [self.np_random.uniform(-1, 1), self.np_random.uniform(-2, -0.5)],
                "life": self.np_random.integers(10, 20),
                "color": (100, 100, 100)
            })

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.2 # Particle gravity
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _get_observation(self):
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Render game elements
        self._render_exit()
        self._render_walls()
        self._render_ghosts()
        self._render_particles()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_walls(self):
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

    def _render_exit(self):
        # Pulsing glow
        pulse = abs(math.sin(self.steps * 0.1)) * 20
        glow_radius = self.TILE_SIZE + int(pulse)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_EXIT_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (self.exit_door.centerx - glow_radius, self.exit_door.centery - glow_radius))

        # Door
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_door)
        door_frame = self.exit_door.inflate(-8, -8)
        pygame.draw.rect(self.screen, self.COLOR_BG, door_frame)

    def _render_player(self):
        # Glow effect
        glow_rect = self.player.inflate(8, 8)
        glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, glow_surf.get_rect(), border_radius=4)
        self.screen.blit(glow_surf, glow_rect.topleft)

        # Player rect
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player, border_radius=2)

    def _render_ghosts(self):
        for i, ghost in enumerate(self.ghosts):
            # Trails
            for j, pos in enumerate(self.ghost_trails[i]):
                alpha = int(255 * (j / len(self.ghost_trails[i])) * 0.3)
                trail_surf = pygame.Surface(pos.size, pygame.SRCALPHA)
                trail_surf.fill((*ghost["color"], alpha))
                self.screen.blit(trail_surf, pos.topleft)
            
            # Ghost body
            alpha = int(128 + math.sin(self.steps * 0.3 + i) * 50) # Flickering alpha
            ghost_surf = pygame.Surface(ghost["rect"].size, pygame.SRCALPHA)
            ghost_surf.fill((*ghost["color"], alpha))
            
            # Eyes
            eye_y = ghost["rect"].height // 3
            eye_w, eye_h = 4, 6
            eye_l_x = ghost["rect"].width // 4
            eye_r_x = ghost["rect"].width - ghost["rect"].width // 4 - eye_w
            pygame.draw.rect(ghost_surf, self.COLOR_PLAYER, (eye_l_x, eye_y, eye_w, eye_h))
            pygame.draw.rect(ghost_surf, self.COLOR_PLAYER, (eye_r_x, eye_y, eye_w, eye_h))

            self.screen.blit(ghost_surf, ghost["rect"].topleft)
    
    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p["life"] * 0.3))
            pygame.draw.rect(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1]), size, size))

    def _render_ui(self):
        # Stage Text
        stage_text = self.font_ui.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (10, 10))

        # Timer Text
        time_left = max(0, self.timer / self.FPS)
        time_color = self.COLOR_TEXT if time_left > 10 else (255, 100, 100)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, time_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Score Text
        score_text = self.font_ui.render(f"SCORE: {int(self.total_score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, 10))

        # Game Over / Win Text
        if self.game_over:
            msg = "GAME OVER"
            color = (200, 50, 50)
        elif self.game_won:
            msg = "YOU ESCAPED!"
            color = (50, 200, 50)
        else:
            return

        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        end_text = self.font_title.render(msg, True, color)
        text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.total_score,
            "steps": self.steps,
            "stage": self.stage,
            "time_left": max(0, self.timer / self.FPS),
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    env = GameEnv()
    env.validate_implementation()
    
    # To run with visualization, comment out the os.environ line and run this:
    # env = GameEnv(render_mode="human") # A proper human render mode would need to be added
    # obs, info = env.reset()
    # running = True
    # while running:
    #     action = env.action_space.sample() # Replace with user input for actual play
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         print(f"Game Over. Score: {info['score']}")
    #         obs, info = env.reset()
    #     # This part is for a manual render loop, not needed for rgb_array
    # env.close()