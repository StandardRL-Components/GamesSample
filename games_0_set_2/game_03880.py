import os
import os
import pygame

os.environ["SDL_VIDEODRIVER"] = "dummy"

import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold Shift to rotate your turret clockwise. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a tank in a top-down arena. Eliminate 5 enemy tanks while dodging their fire. You have 3 health points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_BOUNDARY = (60, 70, 80)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_GLOW = (50, 200, 50, 50)
    COLOR_ENEMY = (220, 50, 50)
    COLOR_ENEMY_GLOW = (220, 50, 50, 50)
    COLOR_PROJECTILE_PLAYER = (255, 255, 255)
    COLOR_PROJECTILE_ENEMY = (255, 150, 150)
    COLOR_EXPLOSION = (255, 200, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (100, 100, 100)

    # Game parameters
    WIDTH, HEIGHT = 640, 400
    MAX_STEPS = 1000
    NUM_ENEMIES = 5
    PLAYER_MAX_HEALTH = 3

    PLAYER_SIZE = 12
    PLAYER_SPEED = 3.0
    PLAYER_FRICTION = 0.92
    TURRET_ROTATION_SPEED = 0.08
    TURRET_LENGTH = 18
    FIRE_COOLDOWN = 10 # frames

    ENEMY_SIZE = 12
    ENEMY_SPEED = 1.0
    ENEMY_BASE_FIRE_RATE = 50
    ENEMY_FIRE_RATE_IMPROVEMENT = 5
    ENEMY_MIN_FIRE_RATE = 10

    PROJECTILE_SIZE = 4
    PROJECTILE_SPEED = 8.0

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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)

        self.player_pos = None
        self.player_vel = None
        self.player_health = None
        self.turret_angle = None
        self.fire_cooldown_timer = None
        self.last_space_held = False

        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.enemy_current_fire_rate = self.ENEMY_BASE_FIRE_RATE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = pygame.Vector2(self.WIDTH / 4, self.HEIGHT / 2)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.turret_angle = 0
        self.fire_cooldown_timer = 0
        self.last_space_held = False
        
        self.enemy_current_fire_rate = self.ENEMY_BASE_FIRE_RATE
        self.enemies = []
        for _ in range(self.NUM_ENEMIES):
            self._spawn_enemy()

        self.projectiles = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Time penalty
        
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

            self._handle_player_input(movement, space_held, shift_held)
            self._update_player()
            reward += self._update_enemies()
            reward += self._update_projectiles()
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            if self.win:
                reward += 100 # Win bonus
            else:
                reward += -100 # Loss penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _spawn_enemy(self):
        # Spawn enemies on the right half of the screen, away from player
        pos = pygame.Vector2(
            self.np_random.uniform(self.WIDTH / 2, self.WIDTH - 30),
            self.np_random.uniform(30, self.HEIGHT - 30)
        )
        self.enemies.append({
            'pos': pos,
            'angle': 0,
            'fire_cooldown': self.np_random.integers(0, self.enemy_current_fire_rate)
        })

    def _handle_player_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: self.player_vel.y -= self.PLAYER_SPEED
        if movement == 2: self.player_vel.y += self.PLAYER_SPEED
        if movement == 3: self.player_vel.x -= self.PLAYER_SPEED
        if movement == 4: self.player_vel.x += self.PLAYER_SPEED

        # Turret rotation
        if shift_held:
            self.turret_angle += self.TURRET_ROTATION_SPEED

        # Firing (on key press, not hold)
        if space_held and not self.last_space_held and self.fire_cooldown_timer <= 0:
            self._fire_projectile(self.player_pos, self.turret_angle, 'player')
            self.fire_cooldown_timer = self.FIRE_COOLDOWN
            # sfx: player_shoot.wav
        self.last_space_held = space_held

    def _update_player(self):
        # Apply velocity and friction
        self.player_pos += self.player_vel
        self.player_vel *= self.PLAYER_FRICTION

        # Clamp position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        if self.fire_cooldown_timer > 0:
            self.fire_cooldown_timer -= 1

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies:
            # AI: Aim at player
            direction_to_player = self.player_pos - enemy['pos']
            enemy['angle'] = direction_to_player.angle_to(pygame.Vector2(1, 0)) * -math.pi / 180.0

            # AI: Move towards player if far, strafe if close
            distance = direction_to_player.length()
            if distance > 150:
                enemy['pos'] += direction_to_player.normalize() * self.ENEMY_SPEED
            elif distance > 0:
                # Strafe by moving perpendicular to player
                perp_vec = direction_to_player.rotate(90).normalize()
                enemy['pos'] += perp_vec * self.ENEMY_SPEED * 0.7
            
            enemy['pos'].x = np.clip(enemy['pos'].x, self.ENEMY_SIZE, self.WIDTH - self.ENEMY_SIZE)
            enemy['pos'].y = np.clip(enemy['pos'].y, self.ENEMY_SIZE, self.HEIGHT - self.ENEMY_SIZE)


            # AI: Fire
            enemy['fire_cooldown'] -= 1
            if enemy['fire_cooldown'] <= 0:
                self._fire_projectile(enemy['pos'], enemy['angle'], 'enemy')
                enemy['fire_cooldown'] = self.enemy_current_fire_rate
                # sfx: enemy_shoot.wav
        return reward

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']

            # Check for out of bounds
            if not (0 < proj['pos'].x < self.WIDTH and 0 < proj['pos'].y < self.HEIGHT):
                self.projectiles.remove(proj)
                continue

            # Check for collisions
            if proj['owner'] == 'player':
                for enemy in self.enemies[:]:
                    if proj['pos'].distance_to(enemy['pos']) < self.ENEMY_SIZE + self.PROJECTILE_SIZE:
                        self.projectiles.remove(proj)
                        self.enemies.remove(enemy)
                        self._create_explosion(enemy['pos'])
                        self.score += 1
                        reward += 10 # Reward for destroying enemy
                        # sfx: enemy_explosion.wav

                        # Increase difficulty
                        self.enemy_current_fire_rate = max(self.ENEMY_MIN_FIRE_RATE, self.enemy_current_fire_rate - self.ENEMY_FIRE_RATE_IMPROVEMENT)
                        break
            elif proj['owner'] == 'enemy':
                if proj['pos'].distance_to(self.player_pos) < self.PLAYER_SIZE + self.PROJECTILE_SIZE:
                    self.projectiles.remove(proj)
                    self.player_health -= 1
                    reward -= 1 # Penalty for taking damage
                    self._create_explosion(self.player_pos, is_player_hit=True)
                    # sfx: player_hit.wav
                    break
        return reward
        
    def _fire_projectile(self, pos, angle, owner):
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.PROJECTILE_SPEED
        self.projectiles.append({'pos': pygame.Vector2(pos), 'vel': vel, 'owner': owner})
        # Muzzle flash
        flash_pos = pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * self.TURRET_LENGTH
        self.particles.append({'pos': flash_pos, 'radius': 8, 'lifetime': 3, 'max_life': 3, 'type': 'flash'})
        
    def _create_explosion(self, pos, is_player_hit=False):
        num_particles = 20
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifetime': lifetime,
                'max_life': lifetime,
                'type': 'explosion'
            })
        if is_player_hit:
            self.particles.append({'pos': pygame.Vector2(self.WIDTH/2, self.HEIGHT/2), 'radius': 0, 'lifetime': 10, 'max_life': 10, 'type': 'screen_flash'})

    def _update_particles(self):
        for p in self.particles[:]:
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
                continue
            if p['type'] == 'explosion':
                p['pos'] += p['vel']
                p['vel'] *= 0.95 # friction
                p['radius'] *= 0.95
            elif p['type'] == 'screen_flash':
                p['radius'] += 1

    def _check_termination(self):
        if self.player_health <= 0:
            self.win = False
            return True
        if not self.enemies:
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.win = False
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Arena boundary
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (10, 10, self.WIDTH - 20, self.HEIGHT - 20), 2)
        
        # Particles
        for p in self.particles:
            life_ratio = p['lifetime'] / p['max_life']
            if p['type'] == 'explosion':
                color = (self.COLOR_EXPLOSION[0], self.COLOR_EXPLOSION[1] * life_ratio, self.COLOR_EXPLOSION[2] * life_ratio)
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color)
            elif p['type'] == 'flash':
                color = (255, 255, 255, int(255 * life_ratio))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius'] * life_ratio), color)
            elif p['type'] == 'screen_flash':
                s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                alpha = int(100 * math.sin(life_ratio * math.pi))
                s.fill((255, 0, 0, alpha))
                self.screen.blit(s, (0,0))


        # Projectiles
        for proj in self.projectiles:
            color = self.COLOR_PROJECTILE_PLAYER if proj['owner'] == 'player' else self.COLOR_PROJECTILE_ENEMY
            pygame.gfxdraw.filled_circle(self.screen, int(proj['pos'].x), int(proj['pos'].y), self.PROJECTILE_SIZE, color)
            pygame.gfxdraw.aacircle(self.screen, int(proj['pos'].x), int(proj['pos'].y), self.PROJECTILE_SIZE, color)

        # Enemies
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'].x), int(enemy['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_SIZE + 4, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_SIZE, self.COLOR_ENEMY)
            
            # Turret
            end_pos_x = enemy['pos'].x + self.TURRET_LENGTH * math.cos(enemy['angle'])
            end_pos_y = enemy['pos'].y + self.TURRET_LENGTH * math.sin(enemy['angle'])
            pygame.draw.line(self.screen, self.COLOR_BOUNDARY, pos_int, (int(end_pos_x), int(end_pos_y)), 3)

        # Player
        if self.player_health > 0:
            pos_int = (int(self.player_pos.x), int(self.player_pos.y))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_SIZE + 5, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
            
            # Turret
            end_pos_x = self.player_pos.x + self.TURRET_LENGTH * math.cos(self.turret_angle)
            end_pos_y = self.player_pos.y + self.TURRET_LENGTH * math.sin(self.turret_angle)
            pygame.draw.line(self.screen, self.COLOR_BOUNDARY, pos_int, (int(end_pos_x), int(end_pos_y)), 3)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 20))

        # Health bar
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (20, 20, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (20, 20, bar_width * health_ratio, 20))

        # Game Over / Win message
        if self.game_over:
            msg = "VICTORY!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ENEMY
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies)
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "dummy" for headless, "x11" or "windows" for visible
    
    # Re-initialize pygame with the new video driver
    pygame.quit()
    pygame.init()

    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Tank Arena")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Game Loop ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game.")
                obs, info = env.reset()
                total_reward = 0

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()