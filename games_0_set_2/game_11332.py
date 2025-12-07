import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:49:10.800903
# Source Brief: brief_01332.md
# Brief Index: 1332
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities to keep the main class clean
class Alien:
    """Represents a single alien enemy."""
    def __init__(self, x, y, speed, fire_rate):
        self.x = x
        self.y = y
        self.speed = speed
        self.fire_rate = fire_rate
        self.fire_cooldown = random.uniform(0, self.fire_rate)
        self.size = 20

class Projectile:
    """Represents a projectile fired by an alien."""
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = 4

class Laser:
    """Represents a player's defensive laser beam."""
    def __init__(self, x, y, height, lifetime):
        self.x = x
        self.y = y
        self.height = height
        self.lifetime = lifetime
        self.initial_lifetime = lifetime

class Particle:
    """Represents a single particle for visual effects."""
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = random.randint(15, 30)
        self.radius = random.uniform(2, 4)

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player defends a base from descending alien attackers
    by firing lasers to deflect their projectiles.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your base from descending aliens. Move your turret and fire a defensive laser beam to deflect incoming projectiles."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to move the turret. Press space to fire a defensive laser beam."
    )
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    MAX_STEPS = 5000

    # Colors
    COLOR_BG = (15, 15, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_BASE = (0, 180, 100)
    COLOR_LASER = (100, 180, 255)
    COLOR_ALIEN = (255, 80, 80)
    COLOR_PROJECTILE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GAMEOVER = (255, 50, 50)
    
    # Player
    PLAYER_SPEED = 6
    PLAYER_LASER_COOLDOWN = 15 # frames
    PLAYER_LASER_HEIGHT = 200
    PLAYER_LASER_LIFETIME = 10 # frames
    
    # Base
    BASE_Y = 360
    
    # Aliens & Difficulty
    INITIAL_ALIEN_SPEED = 0.5
    INITIAL_ALIEN_FIRE_RATE = 120 # frames
    PROJECTILE_SPEED_INCREASE_INTERVAL = 500 # steps
    PROJECTILE_SPEED_INCREASE_AMOUNT = 0.2
    PROJECTILE_FIRE_RATE_INCREASE_INTERVAL = 60 # steps
    PROJECTILE_FIRE_RATE_INCREASE_AMOUNT = 0.995 # multiplier

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_gameover = pygame.font.Font(None, 72)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_x = 0
        self.laser_cooldown_timer = 0
        self.aliens = []
        self.projectiles = []
        self.active_lasers = []
        self.particles = []
        self.stars = []
        self.alien_spawn_timer = 0
        self.current_alien_speed = self.INITIAL_ALIEN_SPEED
        self.current_alien_fire_rate = self.INITIAL_ALIEN_FIRE_RATE
        
        # self.reset() is called by the environment wrapper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_x = self.WIDTH // 2
        self.laser_cooldown_timer = 0
        
        self.aliens.clear()
        self.projectiles.clear()
        self.active_lasers.clear()
        self.particles.clear()

        self.current_alien_speed = self.INITIAL_ALIEN_SPEED
        self.current_alien_fire_rate = self.INITIAL_ALIEN_FIRE_RATE
        self.alien_spawn_timer = 0
        self._spawn_alien()
        
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2))
            for _ in range(150)
        ]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1 # Survival reward

        movement = action[0]
        space_pressed = action[1] == 1
        
        self._handle_input(movement, space_pressed)
        self._update_lasers()
        self._update_aliens()
        deflection_reward = self._update_projectiles()
        reward += deflection_reward
        self._update_particles()
        self._update_difficulty()

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            reward = -100.0
        
        self.score += deflection_reward
        
        truncated = self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed):
        if movement == 3: # Left
            self.player_x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_x += self.PLAYER_SPEED
        self.player_x = np.clip(self.player_x, 0, self.WIDTH)

        self.laser_cooldown_timer = max(0, self.laser_cooldown_timer - 1)
        if space_pressed and self.laser_cooldown_timer == 0:
            self.laser_cooldown_timer = self.PLAYER_LASER_COOLDOWN
            laser_y = self.BASE_Y - 20
            new_laser = Laser(self.player_x, laser_y, self.PLAYER_LASER_HEIGHT, self.PLAYER_LASER_LIFETIME)
            self.active_lasers.append(new_laser)
            # // SFX: Laser fire sound

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.PROJECTILE_SPEED_INCREASE_INTERVAL == 0:
            self.current_alien_speed += self.PROJECTILE_SPEED_INCREASE_AMOUNT
        
        if self.steps > 0 and self.steps % self.PROJECTILE_FIRE_RATE_INCREASE_INTERVAL == 0:
            self.current_alien_fire_rate = max(20, self.current_alien_fire_rate * self.PROJECTILE_FIRE_RATE_INCREASE_AMOUNT)

    def _spawn_alien(self):
        x = random.randint(50, self.WIDTH - 50)
        y = -20
        self.aliens.append(Alien(x, y, self.current_alien_speed, self.current_alien_fire_rate))

    def _update_aliens(self):
        self.alien_spawn_timer -= 1
        if self.alien_spawn_timer <= 0:
            self._spawn_alien()
            self.alien_spawn_timer = random.randint(180, 300)

        for alien in self.aliens[:]:
            alien.y += alien.speed
            if alien.y > self.HEIGHT + alien.size:
                self.aliens.remove(alien)
                continue

            alien.fire_cooldown -= 1
            if alien.fire_cooldown <= 0:
                alien.fire_cooldown = alien.fire_rate
                
                target_x = self.player_x + random.uniform(-100, 100)
                angle_to_target = math.atan2(self.BASE_Y - alien.y, target_x - alien.x)
                
                proj_speed = 2.5 + (self.current_alien_speed - self.INITIAL_ALIEN_SPEED) * 2
                vx = math.cos(angle_to_target) * proj_speed
                vy = math.sin(angle_to_target) * proj_speed

                self.projectiles.append(Projectile(alien.x, alien.y, vx, vy))
                # // SFX: Alien fire sound

    def _update_projectiles(self):
        deflection_reward = 0
        for proj in self.projectiles[:]:
            proj.x += proj.vx
            proj.y += proj.vy

            deflected = False
            for laser in self.active_lasers:
                if (abs(proj.x - laser.x) < 10 and 
                    laser.y - laser.height <= proj.y <= laser.y):
                    
                    deflection_reward += 1.0
                    # // SFX: Deflection sound
                    self._create_particles(proj.x, proj.y, self.COLOR_LASER, 15)

                    proj.vy = -abs(proj.vy) * random.uniform(0.8, 1.2)
                    proj.vx = random.uniform(-2, 2)
                    deflected = True
                    break 
            
            if deflected: continue

            if proj.y >= self.BASE_Y - 5:
                self.game_over = True
                # // SFX: Base explosion sound
                self._create_particles(proj.x, self.BASE_Y, self.COLOR_GAMEOVER, 50)
                self.projectiles.remove(proj)
                continue

            if not (0 < proj.x < self.WIDTH and -10 < proj.y < self.HEIGHT):
                self.projectiles.remove(proj)
        
        return deflection_reward

    def _update_lasers(self):
        for laser in self.active_lasers[:]:
            laser.lifetime -= 1
            if laser.lifetime <= 0:
                self.active_lasers.remove(laser)

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def _update_particles(self):
        for p in self.particles[:]:
            p.x += p.vx
            p.y += p.vy
            p.lifetime -= 1
            p.radius *= 0.97
            if p.lifetime <= 0 or p.radius < 0.5:
                self.particles.remove(p)

    def _check_termination(self):
        return self.game_over
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        for x, y, size in self.stars:
            self.screen.set_at((x, y), (200, 200, 200))

        pygame.draw.line(self.screen, self.COLOR_BASE, (0, self.BASE_Y), (self.WIDTH, self.BASE_Y), 5)
        
        for alien in self.aliens:
            rect = pygame.Rect(alien.x - alien.size / 2, alien.y - alien.size / 2, alien.size, alien.size)
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, rect, border_radius=3)
            pygame.draw.rect(self.screen, (255, 150, 150), rect.inflate(-4, -4), border_radius=2)

        for proj in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj.x), int(proj.y), int(proj.radius) + 2, (*self.COLOR_PROJECTILE, 60))
            pygame.gfxdraw.filled_circle(self.screen, int(proj.x), int(proj.y), int(proj.radius), self.COLOR_PROJECTILE)

        for laser in self.active_lasers:
            alpha = int(255 * (laser.lifetime / laser.initial_lifetime))
            start_pos = (int(laser.x), int(laser.y))
            end_pos = (int(laser.x), int(laser.y - laser.height))
            
            # Draw multiple lines for a glow effect
            pygame.draw.line(self.screen, (*self.COLOR_LASER, int(alpha * 0.3)), (start_pos[0] - 2, start_pos[1]), (end_pos[0] - 2, end_pos[1]), 2)
            pygame.draw.line(self.screen, (*self.COLOR_LASER, int(alpha * 0.3)), (start_pos[0] + 2, start_pos[1]), (end_pos[0] + 2, end_pos[1]), 2)
            pygame.draw.line(self.screen, (*self.COLOR_LASER, alpha), start_pos, end_pos, 2)
        
        turret_poly = [(self.player_x - 12, self.BASE_Y), (self.player_x + 12, self.BASE_Y), (self.player_x + 8, self.BASE_Y - 15), (self.player_x - 8, self.BASE_Y - 15)]
        pygame.gfxdraw.aapolygon(self.screen, turret_poly, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, turret_poly, self.COLOR_PLAYER)

        for p in self.particles:
            alpha = int(255 * (p.lifetime / 30))
            pygame.gfxdraw.filled_circle(self.screen, int(p.x), int(p.y), int(p.radius), (*p.color, alpha))

    def _render_ui(self):
        time_survived = self.steps / self.FPS
        score_text = f"SCORE: {int(self.score)} | TIME: {time_survived:.1f}s"
        text_surface = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surface, (10, 10))

        if self.game_over:
            game_over_text = self.font_gameover.render("GAME OVER", True, self.COLOR_GAMEOVER)
            text_rect = game_over_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(game_over_text, text_rect)
            
            final_score_text = self.font_main.render(f"Final Score: {int(self.score)}", True, self.COLOR_TEXT)
            score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
            self.screen.blit(final_score_text, score_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block is for human play and is not used by the evaluation system.
    # It will not be run by the test suite.
    # You can use it to test your environment locally.
    
    # The following code is a patch to remove the validation call,
    # as it's not part of the standard gym.Env interface and may cause issues
    # with some environment wrappers.
    try:
        del GameEnv.validate_implementation
    except AttributeError:
        pass # Already removed or never existed.

    # Un-dummy the video driver to see the game
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_over = False
    
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Laser Base Defense")
    clock = pygame.time.Clock()

    while running:
        movement, space = 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if not game_over:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            if keys[pygame.K_SPACE]: space = 1
        
        if keys[pygame.K_r] and game_over:
            obs, info = env.reset()
            game_over = False
        if keys[pygame.K_ESCAPE]: running = False
            
        action = [movement, space, 0]

        if not game_over:
            obs, reward, terminated, truncated, info = env.step(action)
            game_over = terminated or truncated

        # Pygame uses (width, height) but our observation is (height, width, 3)
        # Transpose it back for rendering
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(GameEnv.FPS)

    env.close()