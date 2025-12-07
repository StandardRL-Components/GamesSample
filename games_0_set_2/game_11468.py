import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:01:05.238019
# Source Brief: brief_01468.md
# Brief Index: 1468
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Defend your base from waves of attacking aliens. Use your turret to fire quick shots "
        "or powerful charged shots to survive until the timer runs out."
    )
    user_guide = (
        "Controls: ←→ to aim your cannon. Press space to fire a quick shot. "
        "Hold shift to charge and release to fire a powerful stunning shot."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 60

    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_BASE = (80, 80, 120)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_QUICK_SHOT = (100, 255, 255)
    COLOR_CHARGED_SHOT = (255, 255, 100)
    COLOR_STUNNED_ALIEN = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_ENERGY_BAR = (0, 150, 255)
    COLOR_HEALTH_BAR = (255, 80, 80)
    COLOR_CHARGE_BAR = (255, 180, 0)

    # Player
    PLAYER_POS = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30)
    PLAYER_ENERGY_MAX = 100
    PLAYER_ENERGY_REGEN = 0.25  # per step
    PLAYER_ANGLE_SPEED = 0.05  # radians per step
    PLAYER_MIN_ANGLE = -math.pi / 2.2
    PLAYER_MAX_ANGLE = math.pi / 2.2

    # Base
    BASE_HEALTH_MAX = 100

    # Bullets
    QUICK_SHOT_COST = 10
    QUICK_SHOT_SPEED = 8
    QUICK_SHOT_DMG = 10
    CHARGED_SHOT_MIN_CHARGE = 30
    CHARGED_SHOT_SPEED = 12
    CHARGED_SHOT_DMG = 20
    CHARGE_RATE = 2  # per step

    # Aliens
    ALIEN_HEALTH_MAX = 30
    ALIEN_STUN_DURATION = 90  # steps
    ALIEN_REACH_DAMAGE = 25
    INITIAL_SPAWN_RATE = 120  # steps between spawns
    INITIAL_SPEED = 0.8

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
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_timer = pygame.font.SysFont("Consolas", 24, bold=True)

        self.stars = []
        self._generate_stars(200)

        # These attributes are defined here and initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.base_health = 0
        self.player_energy = 0
        self.cannon_angle = 0
        self.charge_level = 0
        self.aliens = []
        self.bullets = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.current_spawn_rate = 0
        self.spawn_timer = 0
        self.current_alien_speed = 0
        self.screen_shake = 0

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.GAME_DURATION_SECONDS * self.FPS
        self.base_health = self.BASE_HEALTH_MAX
        self.player_energy = self.PLAYER_ENERGY_MAX
        self.cannon_angle = 0
        self.charge_level = 0
        self.aliens = []
        self.bullets = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.spawn_timer = self.current_spawn_rate
        self.current_alien_speed = self.INITIAL_SPEED
        self.screen_shake = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        reward = 0
        self.steps += 1
        self.time_remaining -= 1

        reward += self._handle_input(action)
        self._update_game_state()
        reward += self._handle_collisions()

        terminated = self._check_termination()
        truncated = False # Game does not truncate
        if terminated and not self.game_over:
            self.game_over = True
            if self.base_health <= 0:
                reward = -100 # Using a fixed penalty as per brief
            elif self.time_remaining <= 0:
                reward = 100 # Using a fixed bonus as per brief

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # Aiming
        if movement == 3:  # Left
            self.cannon_angle -= self.PLAYER_ANGLE_SPEED
        elif movement == 4:  # Right
            self.cannon_angle += self.PLAYER_ANGLE_SPEED
        self.cannon_angle = np.clip(self.cannon_angle, self.PLAYER_MIN_ANGLE, self.PLAYER_MAX_ANGLE)

        # Quick Shot (on press)
        if space_held and not self.prev_space_held and self.player_energy >= self.QUICK_SHOT_COST:
            self.player_energy -= self.QUICK_SHOT_COST
            vel_x = self.QUICK_SHOT_SPEED * math.sin(self.cannon_angle)
            vel_y = -self.QUICK_SHOT_SPEED * math.cos(self.cannon_angle)
            self.bullets.append({
                'pos': list(self.PLAYER_POS), 'vel': [vel_x, vel_y], 'type': 'quick', 'radius': 4
            })
            # Sound: Pew!

        # Charge Shot
        if shift_held:
            if self.player_energy > 0:
                self.charge_level += self.CHARGE_RATE
                self.player_energy -= self.CHARGE_RATE / 5 # Energy drain while charging
                self.charge_level = min(self.charge_level, self.PLAYER_ENERGY_MAX)
        # Fire charged shot (on release)
        elif not shift_held and self.prev_shift_held:
            if self.charge_level >= self.CHARGED_SHOT_MIN_CHARGE:
                cost = self.charge_level
                if self.player_energy >= cost:
                    self.player_energy -= cost
                    vel_x = self.CHARGED_SHOT_SPEED * math.sin(self.cannon_angle)
                    vel_y = -self.CHARGED_SHOT_SPEED * math.cos(self.cannon_angle)
                    radius = 6 + (self.charge_level / self.PLAYER_ENERGY_MAX) * 6
                    self.bullets.append({
                        'pos': list(self.PLAYER_POS), 'vel': [vel_x, vel_y], 'type': 'charged', 'radius': radius
                    })
                    # Sound: Power-up release!
                    reward += 1 # Reward for stunning, applied on hit
            self.charge_level = 0

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return reward

    def _update_game_state(self):
        # Regenerate player energy
        self.player_energy = min(self.PLAYER_ENERGY_MAX, self.player_energy + self.PLAYER_ENERGY_REGEN)

        # Update difficulty
        self.current_spawn_rate = max(20, self.INITIAL_SPAWN_RATE - self.steps * 0.01 * (self.FPS / 30))
        self.current_alien_speed = self.INITIAL_SPEED + self.steps * 0.005 * (self.FPS / 60)

        # Spawn aliens
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self.spawn_timer = self.current_spawn_rate
            self.aliens.append({
                'pos': [random.uniform(50, self.SCREEN_WIDTH - 50), -20],
                'health': self.ALIEN_HEALTH_MAX,
                'stun_timer': 0,
                'radius': 12
            })
            # Sound: Alien spawn
            
        # Update aliens
        for alien in self.aliens:
            alien['pos'][1] += self.current_alien_speed
            if alien['stun_timer'] > 0:
                alien['stun_timer'] -= 1

        # Update bullets
        for bullet in self.bullets:
            bullet['pos'][0] += bullet['vel'][0]
            bullet['pos'][1] += bullet['vel'][1]

        # Update particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        
        # Update screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 1

        # Cleanup
        self.bullets = [b for b in self.bullets if 0 < b['pos'][0] < self.SCREEN_WIDTH and 0 < b['pos'][1] < self.SCREEN_HEIGHT]
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _handle_collisions(self):
        reward = 0
        
        # Bullet-Alien collisions
        for bullet in self.bullets[:]:
            for alien in self.aliens[:]:
                dist = math.hypot(bullet['pos'][0] - alien['pos'][0], bullet['pos'][1] - alien['pos'][1])
                if dist < bullet['radius'] + alien['radius']:
                    # Sound: Hit
                    if bullet in self.bullets: self.bullets.remove(bullet)

                    is_stunned = alien['stun_timer'] > 0
                    damage = self.QUICK_SHOT_DMG if bullet['type'] == 'quick' else self.CHARGED_SHOT_DMG
                    if is_stunned:
                        damage *= 2
                        reward += 0.2
                    else:
                        reward += 0.1

                    alien['health'] -= damage
                    self._create_explosion(alien['pos'], 10, self.COLOR_ALIEN if not is_stunned else self.COLOR_STUNNED_ALIEN)

                    if bullet['type'] == 'charged' and not is_stunned:
                        alien['stun_timer'] = self.ALIEN_STUN_DURATION
                        reward += 1 # Reward for stunning

                    if alien['health'] <= 0:
                        reward += 5 # Reward for destroying
                        self._create_explosion(alien['pos'], 30, self.COLOR_ALIEN)
                        self.aliens.remove(alien)
                        self.score += 100
                        # Sound: Alien explosion
                    break
        
        # Alien-Base collisions
        for alien in self.aliens[:]:
            if alien['pos'][1] + alien['radius'] > self.PLAYER_POS[1] - 10:
                self.base_health -= self.ALIEN_REACH_DAMAGE
                self.aliens.remove(alien)
                self.screen_shake = 15
                self._create_explosion((alien['pos'][0], self.PLAYER_POS[1] - 10), 40, self.COLOR_BASE)
                # Sound: Base hit / Explosion
        
        self.base_health = max(0, self.base_health)
        return reward

    def _check_termination(self):
        return self.base_health <= 0 or self.time_remaining <= 0

    def _get_observation(self):
        render_offset = [0, 0]
        if self.screen_shake > 0:
            render_offset[0] = random.randint(-4, 4)
            render_offset[1] = random.randint(-4, 4)

        self.screen.fill(self.COLOR_BG)
        self._render_game(render_offset)
        self._render_ui(render_offset)
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self, offset):
        # Stars
        for star in self.stars:
            pos = (star['pos'][0] + offset[0], star['pos'][1] + offset[1])
            pygame.draw.circle(self.screen, star['color'], pos, star['radius'])

        # Base
        base_rect = pygame.Rect(0, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH, 20)
        base_rect.move_ip(offset)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect)
        pygame.draw.rect(self.screen, tuple(c*1.2 for c in self.COLOR_BASE), base_rect, 2)

        # Bullets
        for b in self.bullets:
            pos = (int(b['pos'][0] + offset[0]), int(b['pos'][1] + offset[1]))
            color = self.COLOR_QUICK_SHOT if b['type'] == 'quick' else self.COLOR_CHARGED_SHOT
            self._draw_glow_circle(pos, b['radius'], color)

        # Aliens
        for a in self.aliens:
            pos = (int(a['pos'][0] + offset[0]), int(a['pos'][1] + offset[1]))
            if a['stun_timer'] > 0:
                # Flashing stun effect
                flash_alpha = 128 + 127 * math.sin(self.steps * 0.5)
                self._draw_glow_circle(pos, a['radius'] + 4, (255, 255, 255, flash_alpha))
                self._draw_glow_circle(pos, a['radius'], self.COLOR_STUNNED_ALIEN)
            else:
                self._draw_glow_circle(pos, a['radius'], self.COLOR_ALIEN)

        # Particles
        for p in self.particles:
            p_pos = (int(p['pos'][0] + offset[0]), int(p['pos'][1] + offset[1]))
            alpha = p['life'] / p['max_life']
            color = (p['color'][0], p['color'][1], p['color'][2], int(alpha * 255))
            pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], int(p['radius'] * alpha), color)

        # Player Cannon
        player_x, player_y = self.PLAYER_POS[0] + offset[0], self.PLAYER_POS[1] + offset[1]
        
        # Charge indicator
        if self.charge_level > 0:
            charge_radius = 10 + self.charge_level / self.PLAYER_ENERGY_MAX * 20
            charge_alpha = 50 + (self.charge_level / self.PLAYER_ENERGY_MAX) * 150
            self._draw_glow_circle((player_x, player_y), charge_radius, (*self.COLOR_CHARGED_SHOT, charge_alpha))

        # Cannon barrel
        barrel_length = 30
        end_x = player_x + barrel_length * math.sin(self.cannon_angle)
        end_y = player_y - barrel_length * math.cos(self.cannon_angle)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, (player_x, player_y), (end_x, end_y), 5)
        
        # Cannon base
        self._draw_glow_circle((player_x, player_y), 10, self.COLOR_PLAYER)

    def _render_ui(self, offset):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10 + offset[0], 10 + offset[1]))

        # Time
        time_str = f"{self.time_remaining // self.FPS:02d}"
        time_text = self.font_timer.render(time_str, True, self.COLOR_UI_TEXT)
        time_rect = time_text.get_rect(center=(self.SCREEN_WIDTH // 2 + offset[0], 22 + offset[1]))
        self.screen.blit(time_text, time_rect)

        # Base Health Bar
        health_ratio = self.base_health / self.BASE_HEALTH_MAX
        health_bar_width = (self.SCREEN_WIDTH - 20) * health_ratio
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10 + offset[0], self.SCREEN_HEIGHT - 15 + offset[1], health_bar_width, 10))

        # Player Energy Bar
        energy_ratio = self.player_energy / self.PLAYER_ENERGY_MAX
        energy_bar_width = 150 * energy_ratio
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (self.PLAYER_POS[0] - 75 + offset[0], self.PLAYER_POS[1] + 15 + offset[1], energy_bar_width, 5))
        
        # Player Charge Bar
        if self.charge_level > 0:
            charge_ratio = self.charge_level / self.PLAYER_ENERGY_MAX
            charge_bar_width = 150 * charge_ratio
            pygame.draw.rect(self.screen, self.COLOR_CHARGE_BAR, (self.PLAYER_POS[0] - 75 + offset[0], self.PLAYER_POS[1] + 22 + offset[1], charge_bar_width, 3))


    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(20, 40),
                'max_life': 40,
                'color': color,
                'radius': random.randint(2, 5)
            })
    
    def _generate_stars(self, num_stars):
        for _ in range(num_stars):
            self.stars.append({
                'pos': (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                'radius': random.randint(1, 2),
                'color': random.choice([(100, 100, 100), (150, 150, 150), (200, 200, 200)])
            })

    def _draw_glow_circle(self, pos, radius, color):
        """Draws a circle with a glowing effect."""
        if len(color) == 4: # RGBA
            base_color, alpha = color[:3], color[3]
        else: # RGB
            base_color, alpha = color, 255
        
        radius = int(radius)
        if radius <= 0: return

        # Outer glow layers
        for i in range(radius, 0, -2):
            glow_alpha = int(alpha * (1 - (i / radius))**2 * 0.2)
            if glow_alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], i + radius, (*base_color, glow_alpha))

        # Main circle
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*base_color, alpha))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*base_color, alpha))


    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you might need to remove the SDL_VIDEODRIVER dummy setting
    # at the top of the file.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "quartz" depending on your OS

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Manual Control Test")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # none
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Transpose the observation back for Pygame display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000)
            
        clock.tick(GameEnv.FPS)
        
    env.close()