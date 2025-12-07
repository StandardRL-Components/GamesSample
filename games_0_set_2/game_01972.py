
# Generated: 2025-08-27T18:53:29.852375
# Source Brief: brief_01972.md
# Brief Index: 1972

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper classes for game objects
class Player:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=np.float64)
        self.size = 12
        self.speed = 4.5
        self.lives = 3
        self.fire_cooldown = 0
        self.max_fire_cooldown = 8 # frames
        self.powerup = None
        self.powerup_timer = 0
        self.is_invincible = False
        self.invincibility_timer = 0

class Enemy:
    def __init__(self, pos, enemy_type, wave_modifier):
        self.pos = np.array(pos, dtype=np.float64)
        self.type = enemy_type
        self.wave_modifier = wave_modifier
        self.fire_cooldown = random.randint(30, 90)

        if self.type == "grunt":
            self.size = 10
            self.speed = 1.5 + wave_modifier * 0.2
            self.health = 1
            self.color = (50, 255, 50) # Bright Green
            self.fire_rate_mod = 0.005
        elif self.type == "swooper":
            self.size = 8
            self.speed = 2.0 + wave_modifier * 0.3
            self.health = 1
            self.color = (200, 50, 255) # Bright Purple
            self.fire_rate_mod = 0.008
            self.swoop_phase = random.uniform(0, 2 * math.pi)
        elif self.type == "tank":
            self.size = 15
            self.speed = 1.0 + wave_modifier * 0.15
            self.health = 3
            self.color = (50, 200, 255) # Bright Cyan
            self.fire_rate_mod = 0.012

class Projectile:
    def __init__(self, pos, velocity, color, size):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(velocity, dtype=np.float64)
        self.color = color
        self.size = size
        self.life = 200 # frames

class Particle:
    def __init__(self, pos, velocity, color, size, life):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(velocity, dtype=np.float64)
        self.color = color
        self.size = size
        self.life = life

class PowerUp:
    def __init__(self, pos, powerup_type):
        self.pos = np.array(pos, dtype=np.float64)
        self.type = powerup_type
        self.size = 8
        self.life = 300 # frames

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Press space to fire. Press shift to use a power-up."
    )

    game_description = (
        "A vibrant, procedurally generated top-down space shooter where the player must survive increasingly difficult waves of alien invaders."
    )

    auto_advance = True

    # --- Colors ---
    COLOR_BG = (10, 10, 26)
    COLOR_PLAYER = (220, 220, 255)
    COLOR_PLAYER_GLOW = (100, 100, 255)
    COLOR_PLAYER_PROJECTILE = (0, 150, 255)
    COLOR_ENEMY_PROJECTILE = (255, 80, 80)
    COLOR_POWERUP_SHIELD = (255, 220, 0)
    COLOR_POWERUP_RAPID = (255, 100, 0)
    COLOR_EXPLOSION = [(255, 255, 150), (255, 150, 50), (255, 50, 50)]
    COLOR_TEXT = (255, 255, 255)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.screen_bounds = pygame.Rect(0, 0, self.width, self.height)

        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_big = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self.np_random = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.wave = 0
        self.wave_transition_timer = 0
        self.max_steps = 10000

        self.player = Player(pos=[self.width / 2, self.height - 50])
        
        self.prev_space_held = False
        self.prev_shift_held = False

        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.powerups = []
        self.stars = self._generate_stars(200)

        self._start_next_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01 # Small penalty for surviving to encourage action

        if self.game_over or self.game_won:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Wave Transition ---
        if self.wave_transition_timer > 0:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer == 0:
                self._start_next_wave()
        
        # --- Handle Logic only if not in transition ---
        if self.wave_transition_timer == 0 and not self.game_over and not self.game_won:
            # --- Player Actions ---
            self._handle_player_movement(movement)
            reward += self._handle_player_shooting(space_held)
            self._handle_powerup_activation(shift_held)

            # --- Update Game Objects ---
            self._update_player_state()
            self._update_enemies()
            self._update_projectiles()
            self._update_powerups()
        
        self._update_particles()
        self._update_stars()

        # --- Collision Detection ---
        if self.wave_transition_timer == 0:
            reward += self._handle_collisions()

        # --- Check Game State ---
        if not self.enemies and self.wave_transition_timer == 0 and not self.game_over:
            if self.wave >= 5:
                self.game_won = True
                reward += 500
            else:
                reward += 100
                self.wave_transition_timer = 90 # 3 seconds at 30fps

        self.steps += 1
        terminated = self.player.lives <= 0 or self.game_won or self.steps >= self.max_steps
        if self.player.lives <= 0 and not self.game_over:
            self.game_over = True
            reward -= 10 # Final penalty for game over
            self._create_explosion(self.player.pos, 30, self.player.size * 2)

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_powerups()
        self._render_enemies()
        self._render_projectiles()
        if self.player.lives > 0:
            self._render_player()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player.lives,
            "wave": self.wave,
            "powerup": self.player.powerup
        }

    # --- Game Logic Helpers ---

    def _start_next_wave(self):
        self.wave += 1
        if self.wave > 5: return

        wave_configs = {
            1: {"grunt": 5, "swooper": 0, "tank": 0},
            2: {"grunt": 8, "swooper": 3, "tank": 0},
            3: {"grunt": 10, "swooper": 5, "tank": 1},
            4: {"grunt": 12, "swooper": 8, "tank": 2},
            5: {"grunt": 15, "swooper": 10, "tank": 3},
        }
        config = wave_configs[self.wave]
        
        enemy_list = []
        for type, count in config.items():
            for _ in range(count):
                enemy_list.append(type)
        random.shuffle(enemy_list)

        for i, enemy_type in enumerate(enemy_list):
            x = (i + 1) * (self.width / (len(enemy_list) + 1))
            y = random.randint(40, 100)
            self.enemies.append(Enemy([x, y], enemy_type, self.wave))

    def _handle_player_movement(self, movement):
        direction = np.array([0.0, 0.0])
        if movement == 1: direction[1] -= 1 # Up
        if movement == 2: direction[1] += 1 # Down
        if movement == 3: direction[0] -= 1 # Left
        if movement == 4: direction[0] += 1 # Right

        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            self.player.pos += direction * self.player.speed
            self.player.pos[0] = np.clip(self.player.pos[0], self.player.size, self.width - self.player.size)
            self.player.pos[1] = np.clip(self.player.pos[1], self.player.size, self.height - self.player.size)
            
            # Thruster particles
            if self.steps % 2 == 0:
                p_vel = -direction * 2 + np.array([random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)])
                self.particles.append(Particle(self.player.pos.copy(), p_vel, (200,200,255), random.randint(2,4), 20))


    def _handle_player_shooting(self, space_held):
        fire_pressed = space_held and not self.prev_space_held
        if fire_pressed and self.player.fire_cooldown <= 0:
            # sfx: player_shoot.wav
            vel = np.array([0.0, -10.0])
            self.player_projectiles.append(Projectile(self.player.pos.copy(), vel, self.COLOR_PLAYER_PROJECTILE, 4))
            self.player.fire_cooldown = self.player.max_fire_cooldown
            return 0.0
        return 0.0

    def _handle_powerup_activation(self, shift_held):
        activate_pressed = shift_held and not self.prev_shift_held
        if activate_pressed and self.player.powerup:
            # sfx: powerup_activate.wav
            if self.player.powerup == "shield":
                self.player.is_invincible = True
                self.player.invincibility_timer = 300 # 10 seconds
            elif self.player.powerup == "rapid_fire":
                self.player.powerup_timer = 300 # 10 seconds
            self.player.powerup = None

    def _update_player_state(self):
        if self.player.fire_cooldown > 0:
            self.player.fire_cooldown -= 1
        
        if self.player.invincibility_timer > 0:
            self.player.invincibility_timer -= 1
            if self.player.invincibility_timer == 0:
                self.player.is_invincible = False
        
        if self.player.powerup_timer > 0:
            self.player.powerup_timer -= 1
            self.player.max_fire_cooldown = 3 # Rapid fire
        else:
            self.player.max_fire_cooldown = 8 # Normal fire rate

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement
            if enemy.type == "grunt":
                enemy.pos[1] += enemy.speed * 0.25
            elif enemy.type == "swooper":
                enemy.pos[0] += math.sin(self.steps * 0.02 + enemy.swoop_phase) * enemy.speed * 1.5
                enemy.pos[1] += enemy.speed * 0.3
            elif enemy.type == "tank":
                 enemy.pos[1] += enemy.speed * 0.15

            if enemy.pos[1] > self.height + enemy.size:
                enemy.pos[1] = -enemy.size
                enemy.pos[0] = random.randint(enemy.size, self.width - enemy.size)

            # Firing
            enemy.fire_cooldown -= 1
            fire_chance = (enemy.fire_rate_mod + self.wave * 0.001)
            if enemy.fire_cooldown <= 0 and random.random() < fire_chance:
                # sfx: enemy_shoot.wav
                direction = self.player.pos - enemy.pos
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    vel = direction * 4.0
                    self.enemy_projectiles.append(Projectile(enemy.pos.copy(), vel, self.COLOR_ENEMY_PROJECTILE, 3))
                    enemy.fire_cooldown = random.randint(60, 120) / (1 + self.wave * 0.1)

    def _update_projectiles(self):
        for p in self.player_projectiles[:]:
            p.pos += p.vel
            if not self.screen_bounds.collidepoint(p.pos):
                self.player_projectiles.remove(p)
        for p in self.enemy_projectiles[:]:
            p.pos += p.vel
            if not self.screen_bounds.collidepoint(p.pos):
                self.enemy_projectiles.remove(p)

    def _update_particles(self):
        for p in self.particles[:]:
            p.pos += p.vel
            p.life -= 1
            p.size -= 0.1
            if p.life <= 0 or p.size <= 0:
                self.particles.remove(p)

    def _update_powerups(self):
        for p in self.powerups[:]:
            p.life -= 1
            if p.life <= 0:
                self.powerups.remove(p)

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if np.linalg.norm(proj.pos - enemy.pos) < proj.size + enemy.size:
                    # sfx: enemy_hit.wav
                    self.player_projectiles.remove(proj)
                    enemy.health -= 1
                    reward += 0.1
                    self.score += 10
                    self._create_explosion(proj.pos, 5, 5, (200, 200, 255))
                    if enemy.health <= 0:
                        # sfx: enemy_explode.wav
                        self.enemies.remove(enemy)
                        reward += 1
                        self.score += 100
                        self._create_explosion(enemy.pos, 20, enemy.size * 1.5)
                        if random.random() < 0.15: # 15% chance to drop powerup
                            ptype = random.choice(["shield", "rapid_fire"])
                            self.powerups.append(PowerUp(enemy.pos.copy(), ptype))
                    break
        
        # Enemy projectiles vs Player
        if not self.player.is_invincible:
            for proj in self.enemy_projectiles[:]:
                if np.linalg.norm(proj.pos - self.player.pos) < proj.size + self.player.size:
                    # sfx: player_hit.wav
                    self.enemy_projectiles.remove(proj)
                    self.player.lives -= 1
                    reward -= 10
                    self.player.is_invincible = True
                    self.player.invincibility_timer = 60 # 2 seconds of grace period
                    self._create_explosion(self.player.pos, 15, self.player.size)
                    break
        
        # Player vs Enemies
        if not self.player.is_invincible:
            for enemy in self.enemies[:]:
                if np.linalg.norm(enemy.pos - self.player.pos) < enemy.size + self.player.size:
                    # sfx: player_hit_crash.wav
                    self.enemies.remove(enemy)
                    self.player.lives -= 1
                    reward -= 10
                    self.player.is_invincible = True
                    self.player.invincibility_timer = 60
                    self._create_explosion(self.player.pos, 15, self.player.size)
                    self._create_explosion(enemy.pos, 20, enemy.size * 1.5)
                    break
        
        # Player vs Powerups
        for powerup in self.powerups[:]:
            if np.linalg.norm(powerup.pos - self.player.pos) < powerup.size + self.player.size:
                # sfx: powerup_collect.wav
                if not self.player.powerup: # Only pick up if slot is empty
                    self.player.powerup = powerup.type
                    self.powerups.remove(powerup)
                break
        
        return reward

    # --- Rendering Helpers ---

    def _render_player(self):
        pos = (int(self.player.pos[0]), int(self.player.pos[1]))
        size = int(self.player.size)
        
        # Invincibility flash
        if self.player.invincibility_timer > 0 and self.player.powerup_timer <= 0 and self.steps % 6 < 3:
            return

        # Shield effect
        if self.player.is_invincible:
            if self.player.powerup_timer > 0: # Shield powerup active
                alpha = 100 + int(100 * abs(math.sin(self.steps * 0.1)))
                shield_color = (*self.COLOR_POWERUP_SHIELD, alpha)
            else: # Grace period invincibility
                shield_color = (255, 255, 255, 50)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size + 5, shield_color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size + 5, shield_color)

        # Ship body
        points = [
            (pos[0], pos[1] - size),
            (pos[0] - size * 0.8, pos[1] + size * 0.8),
            (pos[0] + size * 0.8, pos[1] + size * 0.8)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy.pos[0]), int(enemy.pos[1]))
            size = int(enemy.size)
            if enemy.type == "grunt":
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, enemy.color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, enemy.color)
            elif enemy.type == "swooper":
                points = [
                    (pos[0] - size, pos[1] + size),
                    (pos[0], pos[1] - size),
                    (pos[0] + size, pos[1] + size),
                    (pos[0], pos[1] + size * 0.5)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, enemy.color)
                pygame.gfxdraw.filled_polygon(self.screen, points, enemy.color)
            elif enemy.type == "tank":
                rect = pygame.Rect(pos[0] - size, pos[1] - size, size*2, size*2)
                pygame.draw.rect(self.screen, enemy.color, rect, 0, border_radius=3)


    def _render_projectiles(self):
        for p in self.player_projectiles:
            pos = (int(p.pos[0]), int(p.pos[1]))
            tail_pos = (int(p.pos[0] - p.vel[0]*0.5), int(p.pos[1] - p.vel[1]*0.5))
            pygame.draw.line(self.screen, p.color, pos, tail_pos, p.size)
        for p in self.enemy_projectiles:
            pos = (int(p.pos[0]), int(p.pos[1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], p.size, p.color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], p.size, p.color)

    def _render_particles(self):
        for p in self.particles:
            if p.size > 0:
                pos = (int(p.pos[0]), int(p.pos[1]))
                size = int(p.size)
                s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                alpha = max(0, min(255, int(255 * (p.life / 20))))
                color = (*p.color, alpha)
                pygame.draw.circle(s, color, (size, size), size)
                self.screen.blit(s, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_powerups(self):
        for p in self.powerups:
            pos = (int(p.pos[0]), int(p.pos[1]))
            size = int(p.size)
            color = self.COLOR_POWERUP_SHIELD if p.type == "shield" else self.COLOR_POWERUP_RAPID
            
            alpha = 155 + int(100 * math.sin(self.steps * 0.1))
            
            glow_size = size + int(3 * abs(math.sin(self.steps * 0.1)))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_size, (*color, alpha // 4))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_size, (*color, alpha // 2))

            if p.type == "shield":
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
            elif p.type == "rapid_fire":
                rect = pygame.Rect(pos[0] - size, pos[1] - size, size*2, size*2)
                pygame.draw.rect(self.screen, color, rect, 0, border_radius=2)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        wave_text = self.font_main.render(f"WAVE: {self.wave}/5", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.width - wave_text.get_width() - 10, 10))
        
        # Lives
        for i in range(self.player.lives):
            pos = (20 + i * 25, 40)
            points = [
                (pos[0], pos[1] - 8),
                (pos[0] - 6, pos[1] + 6),
                (pos[0] + 6, pos[1] + 6)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        # Powerup indicator
        if self.player.powerup:
            p_pos = (20, 70)
            p_size = 10
            color = self.COLOR_POWERUP_SHIELD if self.player.powerup == "shield" else self.COLOR_POWERUP_RAPID
            if self.player.powerup == "shield":
                pygame.draw.circle(self.screen, color, p_pos, p_size)
            else:
                pygame.draw.rect(self.screen, color, (p_pos[0]-p_size, p_pos[1]-p_size, p_size*2, p_size*2))
        
        # Wave transition text
        if self.wave_transition_timer > 60:
            text = self.font_big.render(f"WAVE {self.wave} CLEARED", True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=(self.width/2, self.height/2 - 20))
            self.screen.blit(text, text_rect)
        elif self.wave_transition_timer > 0:
            text = self.font_big.render(f"WAVE {self.wave+1} INCOMING", True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=(self.width/2, self.height/2 - 20))
            self.screen.blit(text, text_rect)

        # Game over / Win text
        if self.game_over:
            text = self.font_big.render("GAME OVER", True, self.COLOR_ENEMY_PROJECTILE)
            text_rect = text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(text, text_rect)
        elif self.game_won:
            text = self.font_big.render("YOU WIN!", True, self.COLOR_POWERUP_SHIELD)
            text_rect = text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(text, text_rect)


    def _generate_stars(self, n):
        return [
            {
                "pos": [random.randint(0, self.width), random.randint(0, self.height)],
                "speed": random.uniform(0.1, 1.0),
                "color": random.randint(50, 150)
            } for _ in range(n)
        ]

    def _update_stars(self):
        for star in self.stars:
            star["pos"][1] += star["speed"]
            if star["pos"][1] > self.height:
                star["pos"][1] = 0
                star["pos"][0] = random.randint(0, self.width)

    def _render_stars(self):
        for star in self.stars:
            c = star["color"]
            self.screen.set_at((int(star["pos"][0]), int(star["pos"][1])), (c,c,c))
            
    def _create_explosion(self, pos, num_particles, max_speed, color=None):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, max_speed)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed]) * 0.2
            p_color = color if color is not None else random.choice(self.COLOR_EXPLOSION)
            size = random.randint(3, 8)
            life = random.randint(20, 40)
            self.particles.append(Particle(pos.copy(), vel, p_color, size, life))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    # Set to "dummy" to run headless for training, or a real driver for human play.
    # On Linux, you might use "x11". On Windows, "directx". On Mac, "Quartz".
    # Comment the line out to use the default driver.
    # os.environ["SDL_VIDEODRIVER"] = "dummy"

    # To run with keyboard controls for testing, ensure a display is available.
    try:
        env = GameEnv(render_mode="human")
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Galactic Wave")
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Map keyboard to MultiDiscrete action space
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Render to screen
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            clock.tick(30) # 30 FPS

        print(f"Game Over. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
        env.close()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("Could not create display. Running in headless test mode.")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        env = GameEnv()
        env.reset()
        env.step(env.action_space.sample())
        env.close()