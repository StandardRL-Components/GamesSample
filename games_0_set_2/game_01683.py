
# Generated: 2025-08-28T02:22:11.432449
# Source Brief: brief_01683.md
# Brief Index: 1683

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a retro arcade top-down shooter.
    The player must survive for two minutes against waves of descending aliens.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to fire. Survive for 2 minutes."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of descending aliens in this fast-paced retro arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 60
    GAME_DURATION_SECONDS = 120

    # Colors
    COLOR_BG = (10, 10, 20)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_OUTLINE = (128, 255, 200)
    ALIEN_SPECS = {
        'red': {'color': (255, 50, 50), 'health': 1, 'score': 10, 'reward_mult': 1.0, 'speed': 1.5},
        'blue': {'color': (50, 100, 255), 'health': 2, 'score': 20, 'reward_mult': 1.5, 'speed': 1.0},
        'yellow': {'color': (255, 255, 50), 'health': 3, 'score': 30, 'reward_mult': 2.0, 'speed': 0.75},
    }
    COLOR_PROJECTILE_PLAYER = (255, 255, 255)
    COLOR_PROJECTILE_ALIEN = (255, 150, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR = (50, 200, 50)
    COLOR_HEALTH_BAR_BG = (150, 50, 50)

    # Game parameters
    PLAYER_SPEED = 4.5
    PLAYER_FIRE_COOLDOWN = 8  # frames
    PLAYER_HEALTH_MAX = 100
    PLAYER_HIT_DMG = 10
    PLAYER_SIZE = 20
    PROJECTILE_SPEED_PLAYER = 10
    PROJECTILE_SPEED_ALIEN = 5
    PROJECTILE_SIZE = pygame.Rect(0, 0, 4, 10)
    ALIEN_SIZE = 24

    # Difficulty Scaling
    INITIAL_SPAWN_RATE = 0.5
    SPAWN_RATE_INCREASE_PER_SEC = 0.015
    MAX_SPAWN_RATE = 3.0
    INITIAL_ALIEN_FIRE_COOLDOWN = 2.5
    FIRE_COOLDOWN_DECREASE_PER_SEC = 0.018
    MIN_ALIEN_FIRE_COOLDOWN = 0.4

    # Reward structure
    REWARD_SURVIVE_STEP = 0.001
    REWARD_HIT_ALIEN_BASE = 0.5
    REWARD_DESTROY_ALIEN_BASE = 2.0
    REWARD_WIN = 50.0
    REWARD_LOSS = -50.0
    REWARD_WASTED_SHOT = -0.02
    REWARD_TAKE_DAMAGE = -1.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # Initialize state variables to prevent attribute errors before reset
        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_health = 0
        self.player_fire_cooldown = 0
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_timer = 0.0
        self.game_over = False
        self.game_won = False
        self.alien_spawn_accumulator = 0.0
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT - 50)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_fire_cooldown = 0
        
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_timer = 0.0
        self.game_over = False
        self.game_won = False
        self.alien_spawn_accumulator = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        terminated = False

        if self.game_over:
            # If the game is already over, the agent should reset.
            # Return a terminal state with the final reward.
            term_reward = self.REWARD_WIN if self.game_won else self.REWARD_LOSS
            return self._get_observation(), term_reward, True, False, self._get_info()

        self.game_timer += 1 / self.FPS
        if self.game_timer >= self.GAME_DURATION_SECONDS:
            self.game_won = True
            self.game_over = True
            terminated = True
            reward += self.REWARD_WIN
        else:
            reward += self.REWARD_SURVIVE_STEP

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        if space_held and not self.aliens:
            reward += self.REWARD_WASTED_SHOT
        
        self._handle_player_actions(movement, space_held)
        self._update_game_entities()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        if self.player_health <= 0:
            self.player_health = 0
            if not self.game_over: # Ensure loss reward is only applied once
                self.game_over = True
                terminated = True
                reward += self.REWARD_LOSS
                # SFX: Player Explosion
                self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER, 3)

        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_actions(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos.y += self.PLAYER_SPEED
        elif movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos.x += self.PLAYER_SPEED
        
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)

        # Firing
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1
        
        if space_held and self.player_fire_cooldown == 0:
            proj_rect = self.PROJECTILE_SIZE.copy()
            proj_rect.center = (self.player_pos.x, self.player_pos.y - self.PLAYER_SIZE)
            self.player_projectiles.append(proj_rect)
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN
            # SFX: Player Shoot

    def _update_game_entities(self):
        # Update and spawn aliens
        spawn_rate = min(self.MAX_SPAWN_RATE, self.INITIAL_SPAWN_RATE + self.SPAWN_RATE_INCREASE_PER_SEC * self.game_timer)
        self.alien_spawn_accumulator += spawn_rate / self.FPS
        if self.alien_spawn_accumulator >= 1:
            self.alien_spawn_accumulator -= 1
            self._spawn_alien()
        
        fire_cooldown = max(self.MIN_ALIEN_FIRE_COOLDOWN, self.INITIAL_ALIEN_FIRE_COOLDOWN - self.FIRE_COOLDOWN_DECREASE_PER_SEC * self.game_timer)
        for alien in self.aliens[:]:
            spec = self.ALIEN_SPECS[alien['type']]
            alien['rect'].y += spec['speed']
            
            if alien['rect'].top > self.HEIGHT:
                self.aliens.remove(alien)
                continue
            
            alien['fire_timer'] -= 1 / self.FPS
            if alien['fire_timer'] <= 0:
                # SFX: Alien Shoot
                proj_rect = self.PROJECTILE_SIZE.copy()
                proj_rect.center = alien['rect'].center
                self.alien_projectiles.append(proj_rect)
                alien['fire_timer'] = fire_cooldown * self.np_random.uniform(0.8, 1.2)

        # Update projectiles
        self.player_projectiles = [p for p in self.player_projectiles if p.bottom > 0]
        for p in self.player_projectiles: p.y -= self.PROJECTILE_SPEED_PLAYER
            
        self.alien_projectiles = [p for p in self.alien_projectiles if p.top < self.HEIGHT]
        for p in self.alien_projectiles: p.y += self.PROJECTILE_SPEED_ALIEN
        
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)

    def _spawn_alien(self):
        alien_type = self.np_random.choice(list(self.ALIEN_SPECS.keys()), p=[0.5, 0.3, 0.2])
        spec = self.ALIEN_SPECS[alien_type]
        x_pos = self.np_random.uniform(self.ALIEN_SIZE, self.WIDTH - self.ALIEN_SIZE)
        rect = pygame.Rect(x_pos, -self.ALIEN_SIZE, self.ALIEN_SIZE, self.ALIEN_SIZE)
        fire_cooldown = max(self.MIN_ALIEN_FIRE_COOLDOWN, self.INITIAL_ALIEN_FIRE_COOLDOWN - self.FIRE_COOLDOWN_DECREASE_PER_SEC * self.game_timer)
        self.aliens.append({'rect': rect, 'type': alien_type, 'health': spec['health'], 'fire_timer': fire_cooldown * self.np_random.uniform(0.5, 1.5)})

    def _handle_collisions(self):
        reward = 0
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if alien['rect'].colliderect(proj):
                    # SFX: Alien Hit
                    self.player_projectiles.remove(proj)
                    spec = self.ALIEN_SPECS[alien['type']]
                    reward += self.REWARD_HIT_ALIEN_BASE * spec['reward_mult']
                    self._create_explosion(pygame.math.Vector2(proj.center), 5, spec['color'], 0.5)
                    
                    alien['health'] -= 1
                    if alien['health'] <= 0:
                        # SFX: Alien Explosion
                        self.score += spec['score']
                        reward += self.REWARD_DESTROY_ALIEN_BASE * spec['reward_mult']
                        self.aliens.remove(alien)
                        self._create_explosion(pygame.math.Vector2(alien['rect'].center), 20, spec['color'], 1.5)
                    break

        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = self.player_pos
        for proj in self.alien_projectiles[:]:
            if player_rect.colliderect(proj):
                # SFX: Player Hit
                self.alien_projectiles.remove(proj)
                self.player_health -= self.PLAYER_HIT_DMG
                reward += self.REWARD_TAKE_DAMAGE
                self._create_explosion(self.player_pos, 15, self.COLOR_PLAYER_OUTLINE, 1)
                break
        return reward

    def _create_explosion(self, pos, num_particles, color, speed_mult):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'radius': self.np_random.uniform(2, 6), 'life': self.np_random.integers(15, 30), 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color_with_alpha = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius'] * p['life'] / 30), color_with_alpha)

        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.ALIEN_SPECS[alien['type']]['color'], alien['rect'])
            
        for p in self.player_projectiles: pygame.draw.rect(self.screen, self.COLOR_PROJECTILE_PLAYER, p)
        for p in self.alien_projectiles: pygame.draw.rect(self.screen, self.COLOR_PROJECTILE_ALIEN, p)
            
        if self.player_health > 0:
            p, s = self.player_pos, self.PLAYER_SIZE
            points = [(p.x, p.y - s * 0.8), (p.x - s / 2, p.y + s * 0.4), (p.x + s / 2, p.y + s * 0.4)]
            pygame.gfxdraw.aatrigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER_OUTLINE)
            pygame.gfxdraw.filled_trigon(self.screen, int(points[0][0]), int(points[0][1]), int(points[1][0]), int(points[1][1]), int(points[2][0]), int(points[2][1]), self.COLOR_PLAYER)

    def _render_ui(self):
        health_ratio = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, 150, 15))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(150 * health_ratio), 15))
        
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, score_text.get_rect(center=(self.WIDTH / 2, 20)))
        
        time_left = max(0, self.GAME_DURATION_SECONDS - self.game_timer)
        timer_text = self.font_ui.render(f"TIME: {int(time_left // 60):02}:{int(time_left % 60):02}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, timer_text.get_rect(topright=(self.WIDTH - 10, 10)))

        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_HEALTH_BAR if self.game_won else self.ALIEN_SPECS['red']['color']
            end_text = self.font_game_over.render(msg, True, color)
            self.screen.blit(end_text, end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.game_timer, "health": self.player_health}
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    import os
    # This check is to prevent a warning on some systems
    if os.environ.get('SDL_VIDEODRIVER') is None:
        os.environ['SDL_VIDEODRIVER'] = 'x11'

    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Wave Survival - Human Player")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    clock = pygame.time.Clock()

    key_map = {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}
    running = True
    
    while running:
        action = [0, 0, 0]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        moved = False
        for key, move_action in key_map.items():
            if keys[key]:
                action[0] = move_action
                moved = True
                break
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000)
            obs, info = env.reset()

        clock.tick(GameEnv.FPS)

    env.close()