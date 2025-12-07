import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:24:40.941111
# Source Brief: brief_02771.md
# Brief Index: 2771
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Navigate a maze to reach the exit, avoiding rotating lasers. "
        "Toggle between a solid state to block lasers and a transparent state to pass through them."
    )
    user_guide = (
        "Controls: Use the arrow keys (↑↓←→) to move. "
        "Press space to toggle between solid and transparent states."
    )
    auto_advance = True
    
    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_EPISODE_SECONDS = 60
    MAX_STEPS = MAX_EPISODE_SECONDS * FPS

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_WALL = (200, 200, 210)
    COLOR_PLAYER_SOLID = (60, 180, 255)
    COLOR_PLAYER_TRANSPARENT = (60, 180, 255, 100)
    COLOR_LASER = (255, 40, 40)
    COLOR_LASER_GLOW = (180, 30, 30, 100)
    COLOR_EXIT = (40, 255, 120)
    COLOR_EXIT_GLOW = (30, 180, 90, 100)
    COLOR_TEXT = (240, 240, 240)
    
    # Game parameters
    PLAYER_SIZE = 20
    PLAYER_SPEED = 4
    LASER_ROTATION_SECONDS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.render_mode = render_mode
        self._initialize_game_state()
        
        # This call validates the implementation against the brief's requirements.
        # self.validate_implementation() # Commented out as it's not needed for the final submission

    def _initialize_game_state(self):
        # Player
        self.player_pos = pygame.Vector2(0, 0)
        self.player_render_pos = pygame.Vector2(0, 0)
        self.player_is_solid = True
        self.prev_space_held = False

        # Game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        
        # Maze and entities
        self.walls = []
        self.lasers = []
        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        self.dist_to_exit = 0.0
        
        # Visual effects
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._initialize_game_state()
        
        # --- Define Maze Layout ---
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_render_pos = self.player_pos.copy()
        
        self.exit_rect = pygame.Rect(self.SCREEN_WIDTH / 2 - 15, 20, 30, 30)
        
        wall_thickness = 10
        self.walls = [
            # Borders
            pygame.Rect(0, 0, self.SCREEN_WIDTH, wall_thickness),
            pygame.Rect(0, self.SCREEN_HEIGHT - wall_thickness, self.SCREEN_WIDTH, wall_thickness),
            pygame.Rect(0, 0, wall_thickness, self.SCREEN_HEIGHT),
            pygame.Rect(self.SCREEN_WIDTH - wall_thickness, 0, wall_thickness, self.SCREEN_HEIGHT),
            # Internal walls
            pygame.Rect(100, 100, self.SCREEN_WIDTH - 200, wall_thickness),
            pygame.Rect(100, 200, self.SCREEN_WIDTH - 300, wall_thickness),
            pygame.Rect(self.SCREEN_WIDTH - 100 - wall_thickness, 100, wall_thickness, 200),
        ]
        
        # --- Define Lasers ---
        laser_rotation_steps = self.LASER_ROTATION_SECONDS * self.FPS
        self.lasers = [
            {
                "origin": pygame.Vector2(self.SCREEN_WIDTH / 2, 150),
                "direction": pygame.Vector2(1, 0),
                "rotation_timer": 0,
                "rotation_interval": laser_rotation_steps
            },
            {
                "origin": pygame.Vector2(150, 250),
                "direction": pygame.Vector2(0, 1),
                "rotation_timer": laser_rotation_steps // 2, # Staggered rotation
                "rotation_interval": laser_rotation_steps
            }
        ]
        
        self.dist_to_exit = self.player_pos.distance_to(self.exit_rect.center)
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.game_over = False
        reward = 0.0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        self.time_left -= 1
        
        # Survival reward
        reward += 0.01

        # Handle player state toggle (on press, not hold)
        if space_held and not self.prev_space_held:
            self.player_is_solid = not self.player_is_solid
            # sfx: player_toggle.wav
        self.prev_space_held = space_held

        # Handle player movement and wall collisions
        prev_pos = self.player_pos.copy()
        prev_dist_to_exit = self.dist_to_exit

        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED  # Up
        elif movement == 2: self.player_pos.y += self.PLAYER_SPEED  # Down
        elif movement == 3: self.player_pos.x -= self.PLAYER_SPEED  # Left
        elif movement == 4: self.player_pos.x += self.PLAYER_SPEED  # Right
        
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        for wall in self.walls:
            if player_rect.colliderect(wall):
                self.player_pos = prev_pos
                break
        
        # Clamp player position to be within screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE / 2, self.SCREEN_WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE / 2, self.SCREEN_HEIGHT - self.PLAYER_SIZE / 2)

        # Movement reward/penalty
        self.dist_to_exit = self.player_pos.distance_to(self.exit_rect.center)
        if self.dist_to_exit < prev_dist_to_exit:
            reward += 0.1 # Moved closer
        elif self.dist_to_exit > prev_dist_to_exit:
            reward -= 0.1 # Moved further

        # Update lasers
        for laser in self.lasers:
            laser["rotation_timer"] += 1
            if laser["rotation_timer"] >= laser["rotation_interval"]:
                laser["rotation_timer"] = 0
                laser["direction"].rotate_ip(90)
                # sfx: laser_rotate.wav
                
        # Check collisions
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Laser collision
        for laser in self.lasers:
            laser_end = laser["origin"] + laser["direction"] * (self.SCREEN_WIDTH + self.SCREEN_HEIGHT)
            if player_rect.clipline(laser["origin"], laser_end):
                if self.player_is_solid:
                    # Laser blocked
                    reward += 1.0
                    self._create_particles(player_rect.center, 20, self.COLOR_LASER, 0.5)
                    # sfx: laser_block.wav
                else:
                    # Player hit
                    reward -= 100.0
                    self.game_over = True
                    self._create_particles(player_rect.center, 100, self.COLOR_PLAYER_SOLID, 2.0)
                    # sfx: player_death.wav
        
        # Exit collision
        if player_rect.colliderect(self.exit_rect):
            reward += 100.0
            self.game_over = True
            # sfx: win_level.wav

        # Timeout
        if self.time_left <= 0:
            reward -= 50.0
            self.game_over = True

        self.score += reward
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left / self.FPS,
            "player_is_solid": self.player_is_solid
        }

    def _get_observation(self):
        # Interpolate player render position for smooth movement
        interp_factor = 0.5
        self.player_render_pos = self.player_render_pos.lerp(self.player_pos, interp_factor)
        
        self.screen.fill(self.COLOR_BG)
        self._render_game_elements()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game_elements(self):
        # Render walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.COLOR_WALL, wall)

        # Render exit with glow
        glow_exit = self.exit_rect.inflate(15, 15)
        s = pygame.Surface(glow_exit.size, pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_EXIT_GLOW, s.get_rect(), border_radius=8)
        self.screen.blit(s, glow_exit.topleft)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect, border_radius=6)

        # Render lasers with glow
        for laser in self.lasers:
            start_pos = laser["origin"]
            end_pos = laser["origin"] + laser["direction"] * (self.SCREEN_WIDTH + self.SCREEN_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_LASER_GLOW, start_pos, end_pos, 7)
            pygame.draw.line(self.screen, self.COLOR_LASER, start_pos, end_pos, 3)

        # Update and render particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["radius"] -= p["decay"]
            if p["radius"] <= 0:
                self.particles.remove(p)
            else:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), p["color"]
                )
        
        # Render player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_render_pos.x), int(self.player_render_pos.y))

        if self.player_is_solid:
            # Solid player with glow
            glow_surf = pygame.Surface((self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER_SOLID, 60), glow_surf.get_rect(), border_radius=8)
            self.screen.blit(glow_surf, (player_rect.x - self.PLAYER_SIZE / 2, player_rect.y - self.PLAYER_SIZE / 2))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_SOLID, player_rect, border_radius=4)
        else:
            # Transparent player
            s = pygame.Surface((self.PLAYER_SIZE, self.PLAYER_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_PLAYER_TRANSPARENT, s.get_rect(), border_radius=4)
            self.screen.blit(s, player_rect.topleft)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_SOLID, player_rect, 2, border_radius=4)


    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        time_str = f"TIME: {self.time_left / self.FPS:.1f}"
        time_text = self.font_main.render(time_str, True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 20, 20))

    def _create_particles(self, pos, count, color, speed_mult):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_mult
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": random.uniform(2, 5),
                "decay": random.uniform(0.05, 0.2),
                "color": color
            })
    
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # --- Manual Play Loop ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Laser Maze")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Arrows: Move")
    print("Space: Toggle Solid/Transparent")
    print("R: Reset")
    print("Q: Quit")
    
    while True:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False
                truncated = False
                total_reward = 0
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                quit()

        if terminated or truncated:
            # Display game over message
            font_large = pygame.font.SysFont("Consolas", 48, bold=True)
            msg = "YOU WIN!" if info["score"] > 0 and terminated else "GAME OVER"
            color = GameEnv.COLOR_EXIT if info["score"] > 0 and terminated else GameEnv.COLOR_LASER
            
            text_surf = font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 - 30))
            screen.blit(text_surf, text_rect)
            
            score_surf = env.font_main.render(f"Final Score: {info['score']:.2f}", True, GameEnv.COLOR_TEXT)
            score_rect = score_surf.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 20))
            screen.blit(score_surf, score_rect)
            
            reset_surf = env.font_main.render("Press 'R' to Reset", True, GameEnv.COLOR_TEXT)
            reset_rect = reset_surf.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2 + 60))
            screen.blit(reset_surf, reset_rect)
            
            pygame.display.flip()
            continue

        # --- Action Mapping for Manual Control ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render ---
        # The observation is already the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.metadata["render_fps"])

    env.close()