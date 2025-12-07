import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:42:05.971336
# Source Brief: brief_00605.md
# Brief Index: 605
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game objects to keep state organized
class Debris:
    """Represents a piece of orbital debris deployed by the player."""
    def __init__(self, pos):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(random.uniform(-0.4, 0.4), 0)
        self.angle = random.uniform(0, 360)
        self.rot_speed = random.uniform(-0.5, 0.5)
        self.size = (25, 10)
        self.rect = pygame.Rect(0, 0, *self.size)
        self.rect.center = self.pos
        self.trail = [self.pos.copy() for _ in range(10)]

    def update(self):
        self.pos += self.vel
        self.angle = (self.angle + self.rot_speed) % 360
        self.rect.center = self.pos
        
        self.trail.append(self.pos.copy())
        if len(self.trail) > 10:
            self.trail.pop(0)

class Missile:
    """Represents an incoming enemy missile."""
    def __init__(self, pos, target_pos, speed):
        self.pos = pygame.math.Vector2(pos)
        direction = (target_pos - self.pos).normalize()
        self.vel = direction * speed
        self.angle = math.degrees(math.atan2(-direction.x, -direction.y))
        self.size = 12
        self.rect = pygame.Rect(0, 0, self.size, self.size)
        self.rect.center = self.pos
        self.trail = [self.pos.copy() for _ in range(5)]

    def update(self):
        self.pos += self.vel
        self.rect.center = self.pos

        self.trail.append(self.pos.copy())
        if len(self.trail) > 5:
            self.trail.pop(0)

class Explosion:
    """Represents a visual explosion effect."""
    def __init__(self, pos, max_radius, life):
        self.pos = pygame.math.Vector2(pos)
        self.max_radius = max_radius
        self.life = life
        self.max_life = life
        self.radius = 0

    def update(self):
        self.life -= 1
        # Ease-out cubic for radius growth
        progress = 1 - self.life / self.max_life
        self.radius = self.max_radius * (1 - pow(1 - progress, 3))

class Star:
    """Represents a background star for parallax effect."""
    def __init__(self, width, height):
        self.pos = pygame.math.Vector2(random.randint(0, width), random.randint(0, height))
        self.layer = random.choice([1, 2, 3])
        self.speed = self.layer * 0.1
        gray_value = 50 + 50 * self.layer
        self.color = (gray_value, gray_value, gray_value)

    def update(self):
        self.pos.x = (self.pos.x - self.speed) % 640


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend Earth from a barrage of incoming missiles by deploying a limited supply of orbital debris. "
        "Intercept missiles before they reach the planet to survive as long as possible."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the deployment cursor. "
        "Press 'space' to deploy debris in the designated orbital path to intercept missiles."
    )
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.STEPS_PER_SEC = self.MAX_STEPS / 120.0
        
        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_EARTH = (60, 120, 220)
        self.COLOR_EARTH_ATMOSPHERE = (150, 200, 255)
        self.COLOR_DEBRIS = (160, 160, 170)
        self.COLOR_DEBRIS_SHADOW = (130, 130, 140)
        self.COLOR_MISSILE = (255, 80, 80)
        self.COLOR_EXPLOSION = (255, 220, 100)
        self.COLOR_CURSOR = (50, 255, 255)
        self.COLOR_CURSOR_INVALID = (255, 50, 50)
        self.COLOR_UI_TEXT = (220, 220, 220)
        
        # Game parameters
        self.INITIAL_DEBRIS_COUNT = 20
        self.ORBIT_Y = self.HEIGHT * 0.35
        self.ORBIT_DEPLOY_TOLERANCE = 20
        self.EARTH_RADIUS = 80
        self.EARTH_POS = (self.WIDTH // 2, self.HEIGHT + self.EARTH_RADIUS // 2)
        self.CURSOR_SPEED = 8
        self.BASE_MISSILE_SPEED = 1.5
        self.INITIAL_SPAWN_CHANCE = 1.0 / (5.0 * self.STEPS_PER_SEC)
        self.SPAWN_CHANCE_INCREASE_PER_5_STEPS = (0.002 / self.STEPS_PER_SEC) * 5
        self.MISSILE_SPEED_INCREASE_PER_STEP = 0.002
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(None, 28)
            self.font_timer = pygame.font.Font(None, 36)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 22)
            self.font_timer = pygame.font.SysFont("monospace", 30)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.cursor_pos = None
        self.available_debris = 0
        self.missiles = []
        self.debris = []
        self.explosions = []
        self.stars = []
        self.missile_spawn_chance = 0.0
        self.missile_speed_bonus = 0.0
        self.last_space_held = False
        
        self.reset()
        if __name__ != "__main__": # Avoid double-printing when run directly
            pass # validation removed for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.cursor_pos = pygame.math.Vector2(self.WIDTH // 2, self.ORBIT_Y)
        self.available_debris = self.INITIAL_DEBRIS_COUNT
        self.missiles.clear()
        self.debris.clear()
        self.explosions.clear()
        
        self.missile_spawn_chance = self.INITIAL_SPAWN_CHANCE
        self.missile_speed_bonus = 0.0
        self.last_space_held = False
        
        self.stars = [Star(self.WIDTH, self.HEIGHT) for _ in range(150)]
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        self.steps += 1
        
        # --- 1. Handle Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if not self.game_over:
            # Cursor movement
            if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
            elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
            elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
            elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
            
            self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
            self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)
            
            # Debris deployment
            is_deploy_press = space_held and not self.last_space_held
            can_deploy = self.available_debris > 0 and abs(self.cursor_pos.y - self.ORBIT_Y) < self.ORBIT_DEPLOY_TOLERANCE

            if is_deploy_press and can_deploy:
                deploy_pos = (self.cursor_pos.x, self.ORBIT_Y)
                self.debris.append(Debris(deploy_pos))
                self.available_debris -= 1
                # sfx: deployment_sound.play()

        self.last_space_held = space_held
        
        # --- 2. Update Game State ---
        if not self.game_over:
            reward += 0.1 # Survival reward
            
            # Update difficulty
            if self.steps % 5 == 0:
                self.missile_spawn_chance += self.SPAWN_CHANCE_INCREASE_PER_5_STEPS
            if self.steps >= 500:
                self.missile_speed_bonus += self.MISSILE_SPEED_INCREASE_PER_STEP
                
            # Spawn missiles
            if self.np_random.random() < self.missile_spawn_chance:
                spawn_x = self.np_random.uniform(0.1 * self.WIDTH, 0.9 * self.WIDTH)
                spawn_pos = pygame.math.Vector2(spawn_x, -10)
                target_x = self.EARTH_POS[0] + self.np_random.uniform(-self.EARTH_RADIUS * 0.8, self.EARTH_RADIUS * 0.8)
                target_y = self.EARTH_POS[1] - self.EARTH_RADIUS
                target_pos = pygame.math.Vector2(target_x, target_y)
                speed = self.BASE_MISSILE_SPEED + self.missile_speed_bonus
                self.missiles.append(Missile(spawn_pos, target_pos, speed))

            # Update debris
            for d in self.debris:
                d.update()
                if d.pos.x < -d.size[0] or d.pos.x > self.WIDTH + d.size[0]:
                    d.vel.x *= -1 # Bounce off screen edges

            # Update missiles and check collisions
            missiles_to_remove = set()
            debris_to_remove = set()
            for i, m in enumerate(self.missiles):
                m.update()
                
                # Missile-Earth collision
                if m.pos.distance_to(self.EARTH_POS) < self.EARTH_RADIUS:
                    self.game_over = True
                    reward -= 100
                    self.explosions.append(Explosion(m.pos, 80, 60))
                    # sfx: earth_impact_sound.play()
                    break # End game immediately
                
                # Missile-Debris collision
                for j, d in enumerate(self.debris):
                    if m.rect.colliderect(d.rect):
                        reward += 1
                        self.explosions.append(Explosion(m.pos, 40, 30))
                        missiles_to_remove.add(i)
                        debris_to_remove.add(j)
                        # sfx: interception_sound.play()
            
            if self.game_over:
                self.missiles.clear()
            else:
                # Remove destroyed entities
                self.missiles = [m for i, m in enumerate(self.missiles) if i not in missiles_to_remove]
                self.debris = [d for i, d in enumerate(self.debris) if i not in debris_to_remove]

            # Remove off-screen missiles that were missed
            self.missiles = [m for m in self.missiles if m.pos.y < self.HEIGHT + 20]

        # Update explosions
        self.explosions = [e for e in self.explosions if e.life > 0]
        for e in self.explosions:
            e.update()
        
        # Update stars
        for s in self.stars:
            s.update()

        # --- 3. Check Termination ---
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if not self.game_over and self.steps >= self.MAX_STEPS:
            reward += 100 # Victory reward

        self.score += reward
        
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
            "available_debris": self.available_debris,
            "missiles_on_screen": len(self.missiles),
        }

    def _render_game(self):
        # Stars (parallax)
        for s in self.stars:
            pygame.draw.circle(self.screen, s.color, s.pos, s.layer * 0.5)

        # Orbital path
        for i in range(0, self.WIDTH, 20):
            pygame.draw.line(self.screen, (40, 50, 70), (i, self.ORBIT_Y), (i + 10, self.ORBIT_Y))

        # Earth
        pygame.gfxdraw.filled_circle(self.screen, int(self.EARTH_POS[0]), int(self.EARTH_POS[1]), self.EARTH_RADIUS, self.COLOR_EARTH)
        pygame.gfxdraw.aacircle(self.screen, int(self.EARTH_POS[0]), int(self.EARTH_POS[1]), self.EARTH_RADIUS, self.COLOR_EARTH)
        # Atmosphere glow
        for i in range(15, 0, -1):
            alpha = 100 * (1 - i / 15.0)
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(s, int(self.EARTH_POS[0]), int(self.EARTH_POS[1]), self.EARTH_RADIUS + i, (*self.COLOR_EARTH_ATMOSPHERE, int(alpha)))
            self.screen.blit(s, (0,0))
            
        # Debris trails
        for d in self.debris:
            if len(d.trail) > 1:
                trail_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                for i in range(len(d.trail) - 1):
                    alpha = int(100 * (i / len(d.trail)))
                    pygame.draw.line(trail_surf, (*self.COLOR_DEBRIS, alpha), d.trail[i], d.trail[i+1], 1)
                self.screen.blit(trail_surf, (0,0))

        # Missile trails
        for m in self.missiles:
            if len(m.trail) > 1:
                trail_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                for i in range(len(m.trail) - 1):
                    alpha = int(150 * (i / len(m.trail)))
                    start_pos = (int(m.trail[i].x), int(m.trail[i].y))
                    end_pos = (int(m.trail[i+1].x), int(m.trail[i+1].y))
                    pygame.draw.line(trail_surf, (*self.COLOR_MISSILE, alpha), start_pos, end_pos, 2)
                self.screen.blit(trail_surf, (0,0))

        # Debris
        for d in self.debris:
            self._draw_rotated_rect(self.screen, d.rect, d.angle, self.COLOR_DEBRIS, self.COLOR_DEBRIS_SHADOW)

        # Missiles
        for m in self.missiles:
            self._draw_triangle(self.screen, m.pos, m.size, m.angle, self.COLOR_MISSILE)

        # Explosions
        for e in self.explosions:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            alpha = 255 * (e.life / e.max_life)
            
            color_core = (*self.COLOR_EXPLOSION, int(alpha))
            pygame.gfxdraw.filled_circle(s, int(e.pos.x), int(e.pos.y), int(e.radius * 0.5), color_core)
            
            color_glow = (*self.COLOR_EXPLOSION, int(alpha * 0.5))
            pygame.gfxdraw.filled_circle(s, int(e.pos.x), int(e.pos.y), int(e.radius), color_glow)
            pygame.gfxdraw.aacircle(s, int(e.pos.x), int(e.pos.y), int(e.radius), color_glow)
            self.screen.blit(s, (0,0))

        # Cursor
        can_deploy = self.available_debris > 0 and abs(self.cursor_pos.y - self.ORBIT_Y) < self.ORBIT_DEPLOY_TOLERANCE
        cursor_color = self.COLOR_CURSOR if can_deploy else self.COLOR_CURSOR_INVALID
        cursor_y = self.ORBIT_Y if can_deploy else self.cursor_pos.y
        
        pygame.draw.line(self.screen, cursor_color, (self.cursor_pos.x - 10, cursor_y), (self.cursor_pos.x + 10, cursor_y), 2)
        pygame.draw.line(self.screen, cursor_color, (self.cursor_pos.x, cursor_y - 10), (self.cursor_pos.x, cursor_y + 10), 2)
        pygame.gfxdraw.aacircle(self.screen, int(self.cursor_pos.x), int(cursor_y), 12, cursor_color)

    def _render_ui(self):
        # Debris count
        debris_text = self.font_ui.render(f"DEBRIS: {self.available_debris}", True, self.COLOR_UI_TEXT)
        self.screen.blit(debris_text, (15, 15))
        
        # Timer
        time_left = max(0, self.MAX_STEPS - self.steps)
        total_seconds = time_left / self.STEPS_PER_SEC
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        timer_text = self.font_timer.render(f"{minutes:02d}:{seconds:02d}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(timer_text, timer_rect)

    def _draw_rotated_rect(self, surface, rect, angle, color, shadow_color):
        rotated_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(rotated_surface, color, (0, 0, *rect.size), border_radius=2)
        pygame.draw.rect(rotated_surface, shadow_color, (0, rect.height//2, rect.width, rect.height//2), border_radius=2)
        
        rotated_surface = pygame.transform.rotate(rotated_surface, angle)
        new_rect = rotated_surface.get_rect(center=rect.center)
        surface.blit(rotated_surface, new_rect.topleft)
        
    def _draw_triangle(self, surface, pos, size, angle, color):
        points = [
            pygame.math.Vector2(0, -size * 0.75),
            pygame.math.Vector2(-size * 0.5, size * 0.5),
            pygame.math.Vector2(size * 0.5, size * 0.5)
        ]
        rotated_points = [p.rotate(-angle) + pos for p in points]
        int_points = [(int(p.x), int(p.y)) for p in rotated_points]
        pygame.gfxdraw.aapolygon(surface, int_points, color)
        pygame.gfxdraw.filled_polygon(surface, int_points, color)


if __name__ == '__main__':
    # --- Manual Play ---
    # Re-enable display for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Orbital Debris Defense")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while True:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                terminated = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
                
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        if terminated:
            font = pygame.font.Font(None, 72)
            msg = "VICTORY!" if info['steps'] >= env.MAX_STEPS else "GAME OVER"
            color = (100, 255, 100) if msg == "VICTORY!" else (255, 100, 100)
            text = font.render(msg, True, color)
            text_rect = text.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 - 30))
            screen.blit(text, text_rect)
            
            font_small = pygame.font.Font(None, 32)
            text_restart = font_small.render("Press 'R' to Restart", True, (200, 200, 200))
            text_restart_rect = text_restart.get_rect(center=(env.WIDTH/2, env.HEIGHT/2 + 30))
            screen.blit(text_restart, text_restart_rect)


        pygame.display.flip()
        clock.tick(60) # Limit frame rate for manual play