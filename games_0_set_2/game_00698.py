
# Generated: 2025-08-27T14:29:37.521025
# Source Brief: brief_00698.md
# Brief Index: 698

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class Particle:
    """A simple particle class for visual effects."""
    def __init__(self, x, y, color, life, dx, dy, radius):
        self.pos = pygame.Vector2(x, y)
        self.color = color
        self.life = life
        self.max_life = life
        self.vel = pygame.Vector2(dx, dy)
        self.radius = radius

    def update(self):
        self.pos += self.vel
        self.life -= 1
        self.vel *= 0.98  # Damping

    def draw(self, surface, camera_x):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = (*self.color, alpha)
            temp_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (self.radius, self.radius), self.radius)
            surface.blit(temp_surf, (int(self.pos.x - self.radius - camera_x), int(self.pos.y - self.radius)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move, and ↑ to jump. Reach the glowing portal before time runs out."
    )

    game_description = (
        "A fast-paced platformer set in a glowing crystal cavern. Jump across perilous pits to reach the exit."
    )

    auto_advance = True
    
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
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.FPS = 30
        self.TIME_LIMIT_SECONDS = 10
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS
        self.WORLD_WIDTH = self.SCREEN_WIDTH * 4
        
        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_PLAYER = (50, 255, 255)
        self.COLOR_GROUND = (40, 30, 80)
        self.COLOR_CRYSTAL = (200, 220, 255)
        self.COLOR_PIT = (5, 2, 15)
        self.COLOR_EXIT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)

        # Physics
        self.GRAVITY = 0.8
        self.JUMP_STRENGTH = -15
        self.PLAYER_SPEED = 7
        self.PLAYER_DAMPING = 0.85
        self.PLAYER_SIZE = 24
        
        # Fonts
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables (initialized in reset)
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.platforms = []
        self.pits = []
        self.exit_portal = None
        self.particles = []
        self.background_stars = []
        self.camera_x = 0
        self.steps = 0
        self.time_left = 0
        self.score = 0
        self.game_over = False
        self.game_over_message = ""
        self.prev_dist_to_exit = 0
        self.was_over_pit = False

        self.reset()
        self.validate_implementation()

    def _generate_level(self):
        self.platforms.clear()
        self.pits.clear()
        self.background_stars.clear()

        # Generate background stars
        for _ in range(200):
            self.background_stars.append(
                (random.randint(0, self.WORLD_WIDTH), random.randint(0, self.SCREEN_HEIGHT), random.randint(1, 2))
            )

        # Generate platforms and pits
        current_x = 0
        # Start with a safe platform
        self.platforms.append(pygame.Rect(0, self.SCREEN_HEIGHT - 60, 300, 60))
        current_x = 300

        while current_x < self.WORLD_WIDTH - 400:
            platform_width = self.np_random.integers(150, 400)
            pit_width = self.np_random.integers(80, 200)

            self.pits.append(pygame.Rect(current_x, self.SCREEN_HEIGHT - 60, pit_width, 60))
            current_x += pit_width
            
            self.platforms.append(pygame.Rect(current_x, self.SCREEN_HEIGHT - 60, platform_width, 60))
            current_x += platform_width
        
        # Final platform and exit portal
        final_platform_x = self.platforms[-1].right
        self.platforms.append(pygame.Rect(final_platform_x, self.SCREEN_HEIGHT - 60, 400, 60))
        self.exit_portal = pygame.Rect(final_platform_x + 150, self.SCREEN_HEIGHT - 160, 40, 100)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_STEPS
        self.game_over = False
        self.game_over_message = ""

        self.player_pos = pygame.Vector2(150, 200)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = False
        self.particles = []

        self._generate_level()
        self.prev_dist_to_exit = abs(self.player_pos.x - self.exit_portal.centerx)
        self.was_over_pit = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement = action[0]
        reward = 0
        terminated = False

        if not self.game_over:
            self._update_player(movement)
            reward = self._calculate_reward()
            self.score += reward
            self.time_left -= 1
            terminated = self._check_termination()

            if terminated:
                self.game_over = True
                terminal_reward = 0
                if "VICTORY" in self.game_over_message:
                    terminal_reward = 100
                elif "FELL" in self.game_over_message:
                    terminal_reward = -50
                elif "TIME" in self.game_over_message:
                    terminal_reward = -10
                self.score += terminal_reward
                reward += terminal_reward
        
        self._update_particles()
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x *= self.PLAYER_DAMPING

        # Vertical movement (Jump)
        if movement == 1 and self.on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: jump

        # Apply gravity
        self.player_vel.y += self.GRAVITY
        if self.player_vel.y > 20: self.player_vel.y = 20 # Terminal velocity

        # Update position
        self.player_pos += self.player_vel
        self._handle_collisions()

    def _handle_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        self.on_ground = False
        is_over_pit = False

        # Platform collisions
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if self.player_vel.y > 0 and player_rect.bottom > plat.top:
                    player_rect.bottom = plat.top
                    self.player_pos.y = player_rect.y
                    self.player_vel.y = 0
                    self.on_ground = True
        
        # Check if over a pit
        for pit in self.pits:
            if pit.left < player_rect.centerx < pit.right:
                is_over_pit = True
                break

        # Pit jump reward logic
        if self.was_over_pit and self.on_ground:
            self.score += 5.0
            # sfx: success
        self.was_over_pit = is_over_pit

        # Screen boundaries
        if self.player_pos.x < 0:
            self.player_pos.x = 0
            self.player_vel.x = 0
        if self.player_pos.x > self.WORLD_WIDTH - self.PLAYER_SIZE:
            self.player_pos.x = self.WORLD_WIDTH - self.PLAYER_SIZE
            self.player_vel.x = 0

    def _calculate_reward(self):
        dist_to_exit = abs(self.player_pos.x - self.exit_portal.centerx)
        reward = 0
        
        # Reward for moving closer to the exit
        progress = self.prev_dist_to_exit - dist_to_exit
        if progress > 0:
            reward += progress * 0.01 # Scaled down to match brief's magnitude
        else:
            reward += progress * 0.002 # Scaled down
            
        self.prev_dist_to_exit = dist_to_exit
        return reward

    def _check_termination(self):
        # Fell into a pit
        if self.player_pos.y > self.SCREEN_HEIGHT:
            self.game_over_message = "YOU FELL!"
            # sfx: fall
            return True
        # Reached exit
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        if player_rect.colliderect(self.exit_portal):
            self.game_over_message = "VICTORY!"
            # sfx: win
            return True
        # Ran out of time
        if self.time_left <= 0:
            self.game_over_message = "TIME'S UP!"
            # sfx: timeout
            return True
        return False
        
    def _add_particles(self, n, x, y, color, life_range, vel_range, radius_range):
        for _ in range(n):
            life = random.uniform(life_range[0], life_range[1])
            vel = pygame.Vector2(random.uniform(vel_range[0], vel_range[1]), 0)
            vel.rotate_ip(random.uniform(0, 360))
            radius = random.uniform(radius_range[0], radius_range[1])
            self.particles.append(Particle(x, y, color, life, vel.x, vel.y, radius))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()
        
        # Add particles to exit portal
        if self.np_random.random() < 0.7:
             self._add_particles(1, self.exit_portal.centerx, self.exit_portal.centery, self.COLOR_EXIT, (20,40), (0, 2), (1, 4))
    
    def _get_observation(self):
        # Update camera to smoothly follow player
        target_cam_x = self.player_pos.x - self.SCREEN_WIDTH / 2.5
        self.camera_x += (target_cam_x - self.camera_x) * 0.1
        self.camera_x = max(0, min(self.camera_x, self.WORLD_WIDTH - self.SCREEN_WIDTH))

        self.screen.fill(self.COLOR_BG)
        self._render_background(self.screen, self.camera_x)
        self._render_game(self.screen, self.camera_x)
        self._render_ui(self.screen)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, surface, camera_x):
        # Draw parallax stars
        for x, y, z in self.background_stars:
            px = (x - camera_x * (0.1 * z)) % self.WORLD_WIDTH
            if 0 <= px <= self.SCREEN_WIDTH:
                color = int(100 + 100 * (z/2))
                pygame.gfxdraw.pixel(surface, int(px), int(y), (color, color, color, 150))

    def _render_game(self, surface, camera_x):
        # Draw platforms and pits
        for plat in self.platforms:
            pygame.draw.rect(surface, self.COLOR_GROUND, plat.move(-camera_x, 0))
            # Draw crystals on platforms
            for i in range(int(plat.width // 40)):
                cx = plat.left + i * 40 + self.np_random.integers(5, 35)
                cheight = self.np_random.integers(10, 30)
                cwidth = self.np_random.integers(5, 15)
                points = [(cx, plat.top), (cx - cwidth, plat.top), (cx, plat.top - cheight)]
                pygame.gfxdraw.aapolygon(surface, [(p[0] - camera_x, p[1]) for p in points], self.COLOR_CRYSTAL)
                pygame.gfxdraw.filled_polygon(surface, [(p[0] - camera_x, p[1]) for p in points], self.COLOR_CRYSTAL)

        for pit in self.pits:
            pygame.draw.rect(surface, self.COLOR_PIT, pit.move(-camera_x, 0))

        # Draw exit portal
        exit_rect_on_screen = self.exit_portal.move(-camera_x, 0)
        pygame.draw.rect(surface, self.COLOR_EXIT, exit_rect_on_screen, 3)

        # Draw particles
        for p in self.particles:
            p.draw(surface, camera_x)

        # Draw player
        player_rect_on_screen = pygame.Rect(
            int(self.player_pos.x - camera_x), int(self.player_pos.y),
            self.PLAYER_SIZE, self.PLAYER_SIZE
        )
        
        # Player glow effect
        for i in range(4, 0, -1):
            glow_surf = pygame.Surface((self.PLAYER_SIZE + i*8, self.PLAYER_SIZE + i*8), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (*self.COLOR_PLAYER, 20 - i*4), glow_surf.get_rect(), border_radius=8)
            surface.blit(glow_surf, (player_rect_on_screen.x - i*4, player_rect_on_screen.y - i*4))

        pygame.draw.rect(surface, self.COLOR_PLAYER, player_rect_on_screen, border_radius=4)

    def _render_ui(self, surface):
        # Time display
        time_str = f"TIME: {self.time_left / self.FPS:.1f}"
        time_surf = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        surface.blit(time_surf, (10, 10))

        # Score display
        score_str = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_str, True, self.COLOR_TEXT)
        surface.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 10, 10))

        # Game over message
        if self.game_over:
            msg_surf = self.font_game_over.render(self.game_over_message, True, self.COLOR_EXIT)
            surface.blit(msg_surf, msg_surf.get_rect(center=surface.get_rect().center))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "distance_to_exit": self.prev_dist_to_exit
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This block allows a human to play the game.
    # It requires pygame to be installed with display support.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "dummy"
        
        screen = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Crystal Caverns")
        
        obs, info = env.reset()
        done = False
        
        while not done:
            action = [0, 0, 0] # Default no-op
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            if keys[pygame.K_DOWN]:
                action[0] = 2
            if keys[pygame.K_LEFT]:
                action[0] = 3
            if keys[pygame.K_RIGHT]:
                action[0] = 4
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the frame
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            env.clock.tick(env.FPS)
            
    except pygame.error as e:
        print("Pygame display could not be initialized. Running in headless mode.")
        print(f"Error: {e}")
        # Run a simple loop to test the environment logic without display
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

    env.close()