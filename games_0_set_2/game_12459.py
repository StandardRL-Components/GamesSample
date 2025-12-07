import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete, Box
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cyberpunk bounty hunting game.
    The player navigates a vertical cityscape using gravity-flipping mechanics
    to hunt down targets while avoiding patrols.
    """
    game_description = "Navigate a cyberpunk cityscape, flipping gravity to hunt down targets and avoid patrols in this fast-paced action platformer."
    user_guide = "Controls: Use ←→ to move. Press ↑ to jump and ↓ to drop through platforms. Use shift to flip gravity and space to shoot."
    auto_advance = True
    metadata = {"render_modes": ["rgb_array"]}

    # --- CONSTANTS ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors (Cyberpunk Neon)
    COLOR_BG = (15, 15, 25)
    COLOR_BG_ACCENT = (25, 25, 45)
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 50)
    COLOR_TARGET = (255, 180, 0)
    COLOR_TARGET_GLOW = (255, 180, 0, 60)
    COLOR_PATROL = (255, 50, 50)
    COLOR_PATROL_GLOW = (255, 50, 50, 60)
    COLOR_PROJECTILE = (200, 220, 255)
    COLOR_PLATFORM = (40, 40, 60)
    COLOR_PLATFORM_LIT = (100, 100, 255)
    COLOR_WHITE = (240, 240, 240)
    COLOR_UI_TEXT = (200, 200, 220)
    COLOR_HEALTH_BAR = (0, 200, 100)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    
    # Physics & Gameplay
    GRAVITY = 0.4
    FRICTION = -0.08
    PLAYER_ACCEL = 0.6
    PLAYER_JUMP_FORCE = 9.0
    PLAYER_MAX_SPEED_X = 5.0
    PLAYER_MAX_SPEED_Y = 12.0
    PLAYER_HEALTH_MAX = 100
    PROJECTILE_SPEED = 12.0
    MAX_STEPS = 1000
    
    # Cooldowns (in steps)
    SHOOT_COOLDOWN_MAX = 10  # 3 shots per second at 30fps
    GRAVITY_FLIP_COOLDOWN_MAX = 30 # 1 flip per second at 30fps

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_on_ground = False
        self.player_trail = deque(maxlen=10)

        self.gravity_direction = 1  # 1 for down, -1 for up
        self.gravity_flip_cooldown = 0
        self.shoot_cooldown = 0
        
        self.platforms = []
        self.targets = []
        self.patrols = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.bounties_completed = 0
        self.game_over = False
        
        self._last_dist_to_target = float('inf')
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.bounties_completed = 0
        self.game_over = False
        
        self.gravity_direction = 1
        self.gravity_flip_cooldown = 0
        self.shoot_cooldown = 0
        
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_trail.clear()
        
        self.targets.clear()
        self.patrols.clear()
        self.projectiles.clear()
        self.particles.clear()

        # Procedural Generation
        self._generate_platforms()
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self._spawn_bounty()
        self._spawn_patrols()

        nearest_target = self._find_nearest_target()
        if nearest_target:
            self._last_dist_to_target = self.player_pos.distance_to(nearest_target['pos'])
        else:
            self._last_dist_to_target = float('inf')

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        
        # Handle inputs and update cooldowns
        self._handle_input(movement, space_held, shift_held)
        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        self.gravity_flip_cooldown = max(0, self.gravity_flip_cooldown - 1)
        
        # Update game objects
        self._update_player()
        self._update_projectiles()
        self._update_targets()
        self._update_patrols()
        self._update_particles()
        
        # Handle collisions and calculate rewards
        reward += self._handle_collisions()
        
        # Continuous reward for moving towards target
        nearest_target = self._find_nearest_target()
        if nearest_target:
            current_dist = self.player_pos.distance_to(nearest_target['pos'])
            if current_dist < self._last_dist_to_target:
                reward += 0.1
            else:
                reward -= 0.1
            self._last_dist_to_target = current_dist
        
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            self.game_over = True
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- PRIVATE LOGIC METHODS ---

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 3:  # Left
            self.player_vel.x -= self.PLAYER_ACCEL
        elif movement == 4:  # Right
            self.player_vel.x += self.PLAYER_ACCEL
        
        # Jumping (relative to gravity)
        if self.player_on_ground:
            if movement == 1: # Up (against gravity)
                self.player_vel.y = -self.PLAYER_JUMP_FORCE * self.gravity_direction
            elif movement == 2: # Down (with gravity)
                # Pushes through one-way platforms
                self.player_pos.y += 3 * self.gravity_direction
        
        # Shooting
        if space_held and self.shoot_cooldown == 0:
            self._fire_projectile()
            self.shoot_cooldown = self.SHOOT_COOLDOWN_MAX
        
        # Gravity Flip
        if shift_held and self.gravity_flip_cooldown == 0:
            self.gravity_direction *= -1
            self.player_vel.y *= 0.5 # Dampen vertical velocity to prevent overshooting
            self.gravity_flip_cooldown = self.GRAVITY_FLIP_COOLDOWN_MAX
            self._create_particles(self.player_pos, 30, self.COLOR_PLAYER, 2.0, 40)

    def _update_player(self):
        # Apply friction
        self.player_vel.x += self.player_vel.x * self.FRICTION
        if abs(self.player_vel.x) < 0.1: self.player_vel.x = 0
        
        # Apply gravity
        self.player_vel.y += self.GRAVITY * self.gravity_direction
        
        # Clamp velocity
        self.player_vel.x = max(-self.PLAYER_MAX_SPEED_X, min(self.PLAYER_MAX_SPEED_X, self.player_vel.x))
        self.player_vel.y = max(-self.PLAYER_MAX_SPEED_Y, min(self.PLAYER_MAX_SPEED_Y, self.player_vel.y))
        
        # Move and check platform collisions
        self.player_pos.x += self.player_vel.x
        self._handle_platform_collisions('horizontal')
        
        self.player_pos.y += self.player_vel.y
        self.player_on_ground = self._handle_platform_collisions('vertical')
        
        # World wrapping
        if self.player_pos.x < 0: self.player_pos.x = self.SCREEN_WIDTH
        if self.player_pos.x > self.SCREEN_WIDTH: self.player_pos.x = 0
        
        # Update trail
        if self.steps % 2 == 0:
            self.player_trail.append(pygame.math.Vector2(self.player_pos))

    def _handle_platform_collisions(self, direction):
        player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 10, 10, 20)
        on_ground = False
        
        for plat in self.platforms:
            if player_rect.colliderect(plat):
                if direction == 'horizontal':
                    if self.player_vel.x > 0: # Moving right
                        player_rect.right = plat.left
                    elif self.player_vel.x < 0: # Moving left
                        player_rect.left = plat.right
                    self.player_pos.x = player_rect.centerx
                    self.player_vel.x = 0
                elif direction == 'vertical':
                    # Upward gravity
                    if self.gravity_direction == -1:
                        if self.player_vel.y < 0: # Moving up
                            player_rect.top = plat.bottom
                            self.player_vel.y = 0
                            on_ground = True
                    # Downward gravity
                    else:
                        if self.player_vel.y > 0: # Moving down
                            player_rect.bottom = plat.top
                            self.player_vel.y = 0
                            on_ground = True
                    self.player_pos.y = player_rect.centery
        return on_ground

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p['pos'] += p['vel']
            if not self.screen.get_rect().collidepoint(p['pos']):
                self.projectiles.remove(p)

    def _update_targets(self):
        for target in self.targets:
            # Simple AI: move towards player if close, otherwise patrol
            if self.player_pos.distance_to(target['pos']) < 250:
                direction = (self.player_pos - target['pos']).normalize()
            else:
                direction = (target['patrol_end'] - target['pos'])
                if direction.length() < target['speed']:
                    target['patrol_start'], target['patrol_end'] = target['patrol_end'], target['patrol_start']
                    direction = (target['patrol_end'] - target['pos'])
                direction.normalize_ip()

            target['pos'] += direction * target['speed']

    def _update_patrols(self):
        for patrol in self.patrols:
            direction = (patrol['target_pos'] - patrol['pos'])
            if direction.length() < 2:
                patrol['start_pos'], patrol['target_pos'] = patrol['target_pos'], patrol['start_pos']
                direction = (patrol['target_pos'] - patrol['pos']).normalize()
            else:
                direction.normalize_ip()
            patrol['pos'] += direction * 1.5

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 10, 10, 20)

        # Projectile vs Target
        for proj in self.projectiles[:]:
            for target in self.targets[:]:
                if proj['pos'].distance_to(target['pos']) < 15: # 10 radius target + 5 radius proj
                    self.projectiles.remove(proj)
                    target['health'] -= 25
                    reward += 1 # Reward for hitting
                    self._create_particles(target['pos'], 15, self.COLOR_TARGET, 1.5, 20)
                    if target['health'] <= 0:
                        self.targets.remove(target)
                        reward += 50 # Reward for elimination
                        self.bounties_completed += 1
                        self._create_particles(target['pos'], 50, self.COLOR_WHITE, 3.0, 40)
                        self._spawn_bounty() # Spawn next one
                        self._spawn_patrols() # Update patrols
                    break
        
        # Player vs Patrol
        for patrol in self.patrols:
            if player_rect.colliderect(pygame.Rect(patrol['pos'].x - 5, patrol['pos'].y - 5, 10, 10)):
                self.player_health -= 10
                reward -= 5 # Penalty for taking damage
                self._create_particles(self.player_pos, 20, self.COLOR_PATROL, 2.0, 30)
                # Apply knockback
                knockback_vec = (self.player_pos - patrol['pos']).normalize() * 5
                self.player_vel += knockback_vec

        return reward

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            return True
        # Victory condition from brief: "complete all bounty contracts"
        # Since we spawn them one by one, we can set a goal.
        if self.bounties_completed >= 5:
            self.score += 100 # Final bonus
            self.game_over = True
            return True
        return False

    # --- PRIVATE GENERATION METHODS ---

    def _generate_platforms(self):
        self.platforms.clear()
        # Floor and Ceiling
        self.platforms.append(pygame.Rect(0, self.SCREEN_HEIGHT - 10, self.SCREEN_WIDTH, 10))
        self.platforms.append(pygame.Rect(0, 0, self.SCREEN_WIDTH, 10))

        # Random platforms
        for _ in range(10):
            w = self.np_random.integers(80, 200)
            h = 10
            x = self.np_random.integers(0, self.SCREEN_WIDTH - w)
            y = self.np_random.integers(50, self.SCREEN_HEIGHT - 50)
            # Avoid overlap
            new_plat = pygame.Rect(x, y, w, h)
            if not any(new_plat.colliderect(p) for p in self.platforms):
                self.platforms.append(new_plat)
    
    def _spawn_bounty(self):
        if self.bounties_completed >= 5: return

        health = 100 * (1.05 ** self.bounties_completed)
        speed = 1.0 * (1.05 ** self.bounties_completed)
        
        # Find a clear spawn point on a platform
        plat = random.choice(self.platforms[2:]) # Exclude floor/ceiling
        pos = pygame.math.Vector2(plat.centerx, plat.top - 15)
        
        patrol_start = pygame.math.Vector2(plat.left + 15, plat.top - 15)
        patrol_end = pygame.math.Vector2(plat.right - 15, plat.top - 15)

        self.targets.append({
            'pos': pos,
            'health': health,
            'max_health': health,
            'speed': speed,
            'patrol_start': patrol_start,
            'patrol_end': patrol_end
        })

    def _spawn_patrols(self):
        self.patrols.clear()
        num_patrols = 1 + self.bounties_completed // 2
        for _ in range(num_patrols):
            x1 = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            y1 = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            x2 = self.np_random.uniform(50, self.SCREEN_WIDTH - 50)
            y2 = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
            self.patrols.append({
                'pos': pygame.math.Vector2(x1, y1),
                'start_pos': pygame.math.Vector2(x1, y1),
                'target_pos': pygame.math.Vector2(x2, y2)
            })

    def _fire_projectile(self):
        target = self._find_nearest_target()
        if not target: return
        
        direction = (target['pos'] - self.player_pos).normalize()
        pos = self.player_pos + direction * 15 # Spawn away from player
        vel = direction * self.PROJECTILE_SPEED
        self.projectiles.append({'pos': pos, 'vel': vel})
        self._create_particles(self.player_pos, 5, self.COLOR_PROJECTILE, 1.0, 10, direction * 2)

    def _create_particles(self, pos, count, color, speed_scale, lifespan, base_vel=None):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_scale
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            if base_vel:
                vel += base_vel
            self.particles.append({
                'pos': pygame.math.Vector2(pos),
                'vel': vel,
                'lifespan': self.np_random.integers(lifespan // 2, lifespan),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _find_nearest_target(self):
        if not self.targets:
            return None
        return min(self.targets, key=lambda t: self.player_pos.distance_squared_to(t['pos']))
    
    # --- GYM & PYGAME METHODS ---

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "bounties_completed": self.bounties_completed,
            "targets_remaining": len(self.targets)
        }

    def _render_game(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_elements()
        self._render_platforms()
        self._render_particles()
        self._render_projectiles()
        self._render_patrols()
        self._render_targets()
        self._render_player()
        self._render_ui()

    def _render_background_elements(self):
        for i in range(10):
            x = (self.steps * 0.1 * (i+1) + i*100) % (self.SCREEN_WIDTH + 200) - 200
            w = self.np_random.uniform(20, 50)
            h = self.np_random.uniform(50, 250)
            pygame.draw.rect(self.screen, self.COLOR_BG_ACCENT, (x, self.SCREEN_HEIGHT - h, w, h))

    def _render_platforms(self):
        for plat in self.platforms:
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_LIT, plat, 1)

    def _render_player(self):
        # Trail
        for i, pos in enumerate(self.player_trail):
            alpha = int(255 * (i / len(self.player_trail)))
            s = pygame.Surface((10, 20), pygame.SRCALPHA)
            s.fill((*self.COLOR_PLAYER, alpha * 0.5))
            self.screen.blit(s, (int(pos.x - 5), int(pos.y - 10)))

        # Glow effect
        glow_radius = 20
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), glow_radius, self.COLOR_PLAYER_GLOW)
        
        # Player body
        player_rect = pygame.Rect(self.player_pos.x - 5, self.player_pos.y - 10, 10, 20)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=2)
        
        # Gravity indicator
        if self.gravity_direction == 1:
            pygame.draw.polygon(self.screen, self.COLOR_WHITE, [(player_rect.centerx - 4, player_rect.bottom + 2), (player_rect.centerx + 4, player_rect.bottom + 2), (player_rect.centerx, player_rect.bottom + 6)])
        else:
            pygame.draw.polygon(self.screen, self.COLOR_WHITE, [(player_rect.centerx - 4, player_rect.top - 2), (player_rect.centerx + 4, player_rect.top - 2), (player_rect.centerx, player_rect.top - 6)])


    def _render_targets(self):
        for target in self.targets:
            pos = (int(target['pos'].x), int(target['pos'].y))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 15, self.COLOR_TARGET_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 15, self.COLOR_TARGET_GLOW)
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 10, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 10, self.COLOR_TARGET)
            # Health bar
            health_pct = max(0, target['health'] / target['max_health'])
            bar_width = 20
            pygame.draw.rect(self.screen, (80, 50, 0), (pos[0] - bar_width/2, pos[1] - 20, bar_width, 4))
            pygame.draw.rect(self.screen, self.COLOR_TARGET, (pos[0] - bar_width/2, pos[1] - 20, bar_width * health_pct, 4))

    def _render_patrols(self):
        for patrol in self.patrols:
            pos = (int(patrol['pos'].x), int(patrol['pos'].y))
            points = [
                (pos[0], pos[1] - 6),
                (pos[0] + 6, pos[1] + 6),
                (pos[0] - 6, pos[1] + 6)
            ]
            # Glow
            pygame.gfxdraw.filled_trigon(self.screen, *points[0], *points[1], *points[2], self.COLOR_PATROL_GLOW)
            # Body
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PATROL)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PATROL)

    def _render_projectiles(self):
        for p in self.projectiles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            tail_pos = (int(p['pos'].x - p['vel'].x * 0.5), int(p['pos'].y - p['vel'].y * 0.5))
            pygame.draw.line(self.screen, self.COLOR_WHITE, pos, tail_pos, 3)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_PROJECTILE)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 30)) if p['lifespan'] < 30 else 255
            color_with_alpha = (*p['color'][:3], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), color_with_alpha)

    def _render_ui(self):
        # Player Health
        health_pct = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        bar_w, bar_h = 150, 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, bar_w * health_pct, bar_h))
        health_text = self.font_small.render(f"SHIELD", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Bounty Info
        bounty_text = self.font_large.render(f"Bounty {self.bounties_completed+1}/5", True, self.COLOR_UI_TEXT)
        self.screen.blit(bounty_text, (self.SCREEN_WIDTH - bounty_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 45))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and will open a window.
    # The environment itself does not require a display.
    # To run, do: pip install pygame
    # Then: python your_file_name.py
    
    # Re-enable display for manual play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cyberpunk Bounty Hunter")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
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

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward}")
            print(f"Final Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            
        clock.tick(30) # Run at 30 FPS
        
    env.close()