
# Generated: 2025-08-27T21:06:05.532500
# Source Brief: brief_02678.md
# Brief Index: 2678

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold shift for a temporary shield. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship in a top-down shooter, blasting waves of procedurally generated alien invaders to achieve the highest score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Sizing
    WIDTH, HEIGHT = 640, 400
    PLAYER_SIZE = 15
    ALIEN_SIZE = 12
    PROJECTILE_SIZE = 4
    
    # Colors
    COLOR_BG = (10, 10, 26)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_PLAYER_SHIELD = (100, 200, 255)
    COLOR_PLAYER_PROJECTILE = (255, 255, 255)
    COLOR_ALIEN_TYPES = [
        (255, 80, 80),   # Red: Straight
        (80, 150, 255),  # Blue: Sinusoidal
        (255, 255, 80)   # Yellow: Homing
    ]
    COLOR_ENEMY_PROJECTILE = (255, 100, 200)
    COLOR_UI_TEXT = (220, 220, 220)
    
    # Gameplay
    MAX_STEPS = 2000
    PLAYER_SPEED = 4.0
    PLAYER_FIRE_COOLDOWN = 6  # 5 shots per second
    PLAYER_PROJECTILE_SPEED = 10.0
    PLAYER_LIVES = 3
    SHIELD_DURATION = 15 # 0.5 seconds
    SHIELD_COOLDOWN = 90 # 3 seconds
    
    ALIENS_PER_WAVE = 50
    INITIAL_ALIEN_SPEED = 1.0
    INITIAL_ALIEN_FIRE_CHANCE = 0.005 # per frame per alien

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_wave = pygame.font.Font(None, 36)
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_lives = 0
        self.player_fire_cooldown = 0
        self.player_invincibility_timer = 0

        self.shield_timer = 0
        self.shield_cooldown_timer = 0

        self.aliens = []
        self.projectiles = []
        self.particles = []
        self.stars = []
        
        self.score = 0
        self.steps = 0
        self.wave = 0
        self.aliens_destroyed_in_wave = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_lives = self.PLAYER_LIVES
        self.player_fire_cooldown = 0
        self.player_invincibility_timer = 90 # 3 seconds of spawn protection

        self.shield_timer = 0
        self.shield_cooldown_timer = 0
        
        self.aliens.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.aliens_destroyed_in_wave = 0
        self.game_over = False

        self._spawn_stars()
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.02 # Survival penalty to encourage speed
        
        self._handle_input(action)
        self._update_game_state()
        reward += self._handle_collisions()
        
        self.steps += 1
        terminated = self._check_termination()

        if self.aliens_destroyed_in_wave >= self.ALIENS_PER_WAVE:
            reward += 5.0 # Wave clear bonus
            self.wave += 1
            self.aliens_destroyed_in_wave = 0
            self._spawn_wave()

            if self.wave > 1: # For RL, we consider clearing one wave a "win"
                reward += 50.0
                terminated = True
        
        if terminated and self.player_lives <= 0:
            reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
        
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        # Firing
        if space_held and self.player_fire_cooldown <= 0:
            # sfx: player_shoot.wav
            self.projectiles.append({
                "pos": self.player_pos.copy(),
                "vel": pygame.Vector2(0, -self.PLAYER_PROJECTILE_SPEED),
                "owner": "player",
                "color": self.COLOR_PLAYER_PROJECTILE
            })
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN

        # Shield
        if shift_held and self.shield_cooldown_timer <= 0:
            # sfx: shield_activate.wav
            self.shield_timer = self.SHIELD_DURATION
            self.shield_cooldown_timer = self.SHIELD_COOLDOWN

    def _update_game_state(self):
        # Cooldowns and timers
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        if self.player_invincibility_timer > 0: self.player_invincibility_timer -= 1
        if self.shield_timer > 0: self.shield_timer -= 1
        if self.shield_cooldown_timer > 0: self.shield_cooldown_timer -= 1

        # Stars
        for star in self.stars:
            star["pos"].y += star["speed"]
            if star["pos"].y > self.HEIGHT:
                star["pos"].y = 0
                star["pos"].x = self.np_random.integers(0, self.WIDTH)

        # Player engine trail
        if self.steps % 3 == 0:
            self._create_particles(1, self.player_pos + (0, self.PLAYER_SIZE), (100, 100, 100), 10, 1.5)

        # Projectiles
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            if not (0 < p["pos"].y < self.HEIGHT and 0 < p["pos"].x < self.WIDTH):
                self.projectiles.remove(p)
        
        # Aliens
        speed_mod = self.INITIAL_ALIEN_SPEED + (self.wave - 1) * 0.2
        fire_chance_mod = self.INITIAL_ALIEN_FIRE_CHANCE + (self.wave - 1) * 0.01

        for alien in self.aliens[:]:
            alien["timer"] += 1
            # Movement patterns
            if alien["type"] == 0: # Straight
                alien["pos"].y += speed_mod
            elif alien["type"] == 1: # Sinusoidal
                alien["pos"].y += speed_mod * 0.8
                alien["pos"].x = alien["start_x"] + math.sin(alien["timer"] * 0.05) * 50
            elif alien["type"] == 2: # Homing
                direction = (self.player_pos - alien["pos"]).normalize()
                alien["pos"] += direction * speed_mod * 0.7

            # Firing
            if self.np_random.random() < fire_chance_mod:
                # sfx: enemy_shoot.wav
                proj_vel = (self.player_pos - alien["pos"]).normalize() * (self.PLAYER_PROJECTILE_SPEED / 2)
                self.projectiles.append({
                    "pos": alien["pos"].copy(),
                    "vel": proj_vel,
                    "owner": "enemy",
                    "color": self.COLOR_ENEMY_PROJECTILE
                })

            if alien["pos"].y > self.HEIGHT + self.ALIEN_SIZE:
                self.aliens.remove(alien)

        # Particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for p in self.projectiles[:]:
            if p["owner"] == "player":
                for alien in self.aliens[:]:
                    if p["pos"].distance_to(alien["pos"]) < self.ALIEN_SIZE:
                        # sfx: explosion.wav
                        reward += 0.1 # Hit bonus
                        reward += 1.0 # Destroy bonus
                        self.score += 10
                        self.aliens_destroyed_in_wave += 1
                        
                        self._create_particles(20, alien["pos"], self.COLOR_ALIEN_TYPES[alien["type"]], 30, 3.0)
                        
                        self.aliens.remove(alien)
                        if p in self.projectiles: self.projectiles.remove(p)
                        break

        # Enemy projectiles/aliens vs Player
        is_hit = False
        if self.player_invincibility_timer <= 0 and self.shield_timer <= 0:
            # Check enemy projectiles
            for p in self.projectiles[:]:
                if p["owner"] == "enemy" and p["pos"].distance_to(self.player_pos) < self.PLAYER_SIZE:
                    is_hit = True
                    self.projectiles.remove(p)
                    break
            # Check direct collision with aliens
            if not is_hit:
                for alien in self.aliens[:]:
                    if alien["pos"].distance_to(self.player_pos) < self.PLAYER_SIZE + self.ALIEN_SIZE / 2:
                        is_hit = True
                        self._create_particles(20, alien["pos"], self.COLOR_ALIEN_TYPES[alien["type"]], 30, 3.0)
                        self.aliens.remove(alien)
                        self.aliens_destroyed_in_wave += 1
                        break

        if is_hit:
            # sfx: player_hit.wav
            self.player_lives -= 1
            self.player_invincibility_timer = 90 # 3s invincibility
            self._create_particles(40, self.player_pos, self.COLOR_PLAYER, 40, 4.0)
            self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50) # Reset position
            if self.player_lives <= 0:
                self.game_over = True
        
        return reward
    
    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            pygame.draw.circle(self.screen, star["color"], star["pos"], star["size"])

        # Particles
        for p in self.particles:
            alpha = max(0, 255 * (p["lifespan"] / p["max_lifespan"]))
            color = p["color"]
            s = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (color[0], color[1], color[2], alpha), (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(s, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])))

        # Projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), self.PROJECTILE_SIZE // 2)

        # Aliens
        for alien in self.aliens:
            color = self.COLOR_ALIEN_TYPES[alien["type"]]
            pos = (int(alien["pos"].x), int(alien["pos"].y))
            size = self.ALIEN_SIZE
            if alien["type"] == 0: # Triangle
                points = [(pos[0], pos[1] - size), (pos[0] - size, pos[1] + size), (pos[0] + size, pos[1] + size)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif alien["type"] == 1: # Diamond
                points = [(pos[0], pos[1] - size), (pos[0] + size, pos[1]), (pos[0], pos[1] + size), (pos[0] - size, pos[1])]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            else: # Pentagon
                points = []
                for i in range(5):
                    angle = math.radians(90 + i * 72)
                    points.append((pos[0] + size * math.cos(angle), pos[1] + size * math.sin(angle)))
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Player
        is_invincible = self.player_invincibility_timer > 0
        if not (is_invincible and (self.steps // 4) % 2 == 0): # Blink when invincible
            p1 = (self.player_pos.x, self.player_pos.y - self.PLAYER_SIZE)
            p2 = (self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y + self.PLAYER_SIZE / 2)
            p3 = (self.player_pos.x + self.PLAYER_SIZE / 2, self.player_pos.y + self.PLAYER_SIZE / 2)
            
            # Glow effect
            glow_surf = pygame.Surface((self.PLAYER_SIZE * 3, self.PLAYER_SIZE * 3), pygame.SRCALPHA)
            glow_points = [
                (self.PLAYER_SIZE*1.5, self.PLAYER_SIZE*1.5 - self.PLAYER_SIZE*1.2),
                (self.PLAYER_SIZE*1.5 - self.PLAYER_SIZE*0.8, self.PLAYER_SIZE*1.5 + self.PLAYER_SIZE*0.8),
                (self.PLAYER_SIZE*1.5 + self.PLAYER_SIZE*0.8, self.PLAYER_SIZE*1.5 + self.PLAYER_SIZE*0.8)
            ]
            pygame.gfxdraw.filled_polygon(glow_surf, glow_points, self.COLOR_PLAYER_GLOW)
            self.screen.blit(glow_surf, (self.player_pos.x - self.PLAYER_SIZE*1.5, self.player_pos.y - self.PLAYER_SIZE*1.5))

            # Main ship
            pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

        # Shield
        if self.shield_timer > 0:
            alpha = 100 + 100 * math.sin(self.steps * 0.5) # Pulsing effect
            radius = self.PLAYER_SIZE * 1.5
            s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(s, int(radius), int(radius), int(radius), (*self.COLOR_PLAYER_SHIELD, int(alpha)))
            pygame.gfxdraw.filled_circle(s, int(radius), int(radius), int(radius), (*self.COLOR_PLAYER_SHIELD, int(alpha/2)))
            self.screen.blit(s, (self.player_pos.x - radius, self.player_pos.y - radius))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Lives
        for i in range(self.player_lives):
            p1 = (20 + i * (self.PLAYER_SIZE + 5), 10 + self.PLAYER_SIZE)
            p2 = (20 + i * (self.PLAYER_SIZE + 5) - self.PLAYER_SIZE/2, 10)
            p3 = (20 + i * (self.PLAYER_SIZE + 5) + self.PLAYER_SIZE/2, 10)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [p1,p2,p3], width=0)

        # Wave
        wave_text = self.font_wave.render(f"WAVE {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH/2 - wave_text.get_width()/2, self.HEIGHT - 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.wave,
        }

    def _spawn_stars(self):
        self.stars.clear()
        for _ in range(150):
            self.stars.append({
                "pos": pygame.Vector2(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                "speed": self.np_random.random() * 1.5 + 0.5,
                "size": self.np_random.integers(1, 3),
                "color": random.choice([(100,100,100), (150,150,150), (200,200,200)])
            })

    def _spawn_wave(self):
        self.aliens.clear()
        self.projectiles = [p for p in self.projectiles if p["owner"] == "player"] # Keep player projectiles
        
        rows = 5
        cols = self.ALIENS_PER_WAVE // rows
        for i in range(self.ALIENS_PER_WAVE):
            row = i // cols
            col = i % cols
            
            start_x = (self.WIDTH / (cols + 1)) * (col + 1)
            start_y = -30 - row * (self.ALIEN_SIZE * 3)

            self.aliens.append({
                "pos": pygame.Vector2(start_x, start_y),
                "start_x": start_x,
                "type": self.np_random.integers(0, len(self.COLOR_ALIEN_TYPES)),
                "timer": self.np_random.integers(0, 100), # Stagger movement patterns
            })

    def _create_particles(self, count, pos, color, max_lifespan, max_speed):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * max_speed
            lifespan = self.np_random.integers(max_lifespan // 2, max_lifespan)
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "lifespan": lifespan,
                "max_lifespan": lifespan,
                "color": color,
                "radius": self.np_random.integers(2, 5)
            })

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Spaceship Shooter")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before restarting

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Match the intended framerate
        
    env.close()