
# Generated: 2025-08-27T21:27:42.281440
# Source Brief: brief_02796.md
# Brief Index: 2796

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ↑↓←→ to move. Survive the onslaught!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a 60-second monster onslaught in this top-down arcade game, dodging enemies and grabbing power-ups to boost your score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 5
        self.MONSTER_SIZE = 22
        self.MONSTER_BASE_SPEED = 2
        self.POWERUP_SIZE = 18
        self.INITIAL_LIVES = 3
        self.INVINCIBILITY_DURATION = 5 * self.FPS # 5 seconds

        # --- Colors (Neon on Dark) ---
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_GRID = (30, 10, 50)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_INVINCIBLE = (255, 255, 100)
        self.COLOR_MONSTER_RED = (255, 50, 50)
        self.COLOR_MONSTER_BLUE = (50, 150, 255)
        self.COLOR_MONSTER_YELLOW = (255, 255, 0)
        self.COLOR_POWERUP = (255, 255, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TIMER = (255, 200, 0)
        
        self.MONSTER_COLORS = [self.COLOR_MONSTER_RED, self.COLOR_MONSTER_BLUE, self.COLOR_MONSTER_YELLOW]

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # --- Game State ---
        self.np_random = None
        self.player_rect = None
        self.player_lives = 0
        self.invincibility_timer = 0
        self.monsters = []
        self.powerups = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.monster_spawn_timer = 0
        self.powerup_spawn_timer = 0
        self.monster_spawn_interval = 0
        self.game_area = pygame.Rect(20, 50, self.WIDTH - 40, self.HEIGHT - 70)

        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.player_rect = pygame.Rect(
            self.WIDTH // 2 - self.PLAYER_SIZE // 2,
            self.HEIGHT // 2 - self.PLAYER_SIZE // 2,
            self.PLAYER_SIZE,
            self.PLAYER_SIZE,
        )
        self.player_lives = self.INITIAL_LIVES
        self.invincibility_timer = 0

        self.monsters = []
        self.powerups = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.monster_spawn_interval = 2 * self.FPS
        self.monster_spawn_timer = self.monster_spawn_interval
        self.powerup_spawn_timer = self.np_random.integers(7, 12) * self.FPS
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.01  # Survival reward per step

        self._update_timers()
        self._handle_input(action)
        self._update_monsters()
        self._update_particles()
        self._spawn_entities()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward

        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.player_lives > 0:
                reward += 100  # Survival victory
            else:
                reward -= 100  # Death
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _update_timers(self):
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        
        self.monster_spawn_timer -= 1
        self.powerup_spawn_timer -= 1

        # Difficulty scaling: monster spawn rate increases every 10 seconds
        ten_second_intervals = self.steps // (10 * self.FPS)
        # Decrease interval by 0.2s every 10s, with a minimum of 0.5s
        self.monster_spawn_interval = max(0.5 * self.FPS, (2.0 - ten_second_intervals * 0.2) * self.FPS)

    def _handle_input(self, action):
        movement = action[0]
        dx, dy = 0, 0
        if movement == 1: dy = -self.PLAYER_SPEED  # Up
        elif movement == 2: dy = self.PLAYER_SPEED   # Down
        elif movement == 3: dx = -self.PLAYER_SPEED # Left
        elif movement == 4: dx = self.PLAYER_SPEED  # Right
        
        self.player_rect.move_ip(dx, dy)
        self.player_rect.clamp_ip(self.game_area)

    def _update_monsters(self):
        for monster in self.monsters:
            monster['rect'].move_ip(monster['vel'])
            # Bounce off game area walls
            if monster['rect'].left < self.game_area.left or monster['rect'].right > self.game_area.right:
                monster['vel'][0] *= -1
            if monster['rect'].top < self.game_area.top or monster['rect'].bottom > self.game_area.bottom:
                monster['vel'][1] *= -1
            monster['rect'].clamp_ip(self.game_area)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _spawn_entities(self):
        # Spawn monster
        if self.monster_spawn_timer <= 0:
            self._spawn_monster()
            self.monster_spawn_timer = self.monster_spawn_interval
        
        # Spawn powerup
        if self.powerup_spawn_timer <= 0:
            self._spawn_powerup()
            self.powerup_spawn_timer = self.np_random.integers(7, 12) * self.FPS

    def _get_safe_spawn_pos(self, size):
        while True:
            x = self.np_random.integers(self.game_area.left, self.game_area.right - size)
            y = self.np_random.integers(self.game_area.top, self.game_area.bottom - size)
            spawn_rect = pygame.Rect(x, y, size, size)
            if not self.player_rect.colliderect(spawn_rect.inflate(80, 80)):
                return x, y

    def _spawn_monster(self):
        x, y = self._get_safe_spawn_pos(self.MONSTER_SIZE)
        rect = pygame.Rect(x, y, self.MONSTER_SIZE, self.MONSTER_SIZE)
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.MONSTER_BASE_SPEED + self.np_random.uniform(-0.5, 1.0)
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        
        color_index = self.np_random.integers(0, len(self.MONSTER_COLORS))
        color = self.MONSTER_COLORS[color_index]
        
        self.monsters.append({'rect': rect, 'vel': vel, 'color': color})

    def _spawn_powerup(self):
        if len(self.powerups) < 2: # Limit powerups on screen
            x, y = self._get_safe_spawn_pos(self.POWERUP_SIZE)
            rect = pygame.Rect(x, y, self.POWERUP_SIZE, self.POWERUP_SIZE)
            self.powerups.append(rect)

    def _handle_collisions(self):
        reward = 0
        
        # Player-Monster
        if self.invincibility_timer <= 0:
            collided_monster_idx = self.player_rect.collidelist([m['rect'] for m in self.monsters])
            if collided_monster_idx != -1:
                self.player_lives -= 1
                self.invincibility_timer = 2 * self.FPS # Brief invincibility after hit
                self._create_particle_burst(self.player_rect.center, self.COLOR_PLAYER, 30)
                # sound_effect: player_hit.wav
                if self.player_lives <= 0:
                    self.game_over = True
                
                # Remove the monster that hit the player
                collided_monster = self.monsters.pop(collided_monster_idx)
                self._create_particle_burst(collided_monster['rect'].center, collided_monster['color'], 20)


        # Player-Powerup
        collided_powerup_idx = self.player_rect.collidelist(self.powerups)
        if collided_powerup_idx != -1:
            self.powerups.pop(collided_powerup_idx)
            self.invincibility_timer = self.INVINCIBILITY_DURATION
            self.score += 10
            reward += 1.0
            self._create_particle_burst(self.player_rect.center, self.COLOR_POWERUP, 50)
            # sound_effect: powerup_get.wav
            
        return reward

    def _check_termination(self):
        return self.player_lives <= 0 or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT), 1)
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y), 1)
        pygame.draw.rect(self.screen, self.COLOR_GRID, self.game_area, 2)

    def _render_game(self):
        # Particles
        for p in self.particles:
            size = max(1, int(p['life'] * 0.2))
            pygame.draw.rect(self.screen, p['color'], (p['pos'][0], p['pos'][1], size, size))

        # Powerups
        for powerup_rect in self.powerups:
            self._draw_star(self.screen, self.COLOR_POWERUP, powerup_rect.center, self.POWERUP_SIZE // 2)
            self._draw_glow(self.screen, self.COLOR_POWERUP, powerup_rect.center, 20, 100)

        # Monsters
        for monster in self.monsters:
            pos = (int(monster['rect'].centerx), int(monster['rect'].centery))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.MONSTER_SIZE // 2, monster['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.MONSTER_SIZE // 2, monster['color'])
            self._draw_glow(self.screen, monster['color'], pos, 20, 80)
            
        # Player
        is_invincible = self.invincibility_timer > 0
        player_color = self.COLOR_PLAYER
        if is_invincible:
            # Flash when invincible
            if (self.invincibility_timer > 2 * self.FPS) or (self.steps % 10 < 5):
                 player_color = self.COLOR_PLAYER_INVINCIBLE
            self._draw_glow(self.screen, self.COLOR_PLAYER_INVINCIBLE, self.player_rect.center, 30, 150)

        pygame.draw.rect(self.screen, player_color, self.player_rect)
        self._draw_glow(self.screen, player_color, self.player_rect.center, 25, 120)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 15))

        # Lives
        lives_text = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 15))
        for i in range(self.player_lives):
            life_rect = pygame.Rect(self.WIDTH - 70 + (i * 20), 18, 15, 15)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, life_rect)

        # Timer
        time_left = max(0, self.GAME_DURATION_SECONDS - (self.steps / self.FPS))
        timer_text = self.font_large.render(f"{time_left:.1f}", True, self.COLOR_TIMER)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, 5))

    def _create_particle_burst(self, position, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(position), 'vel': vel, 'life': life, 'color': color})

    def _draw_glow(self, surface, color, center, max_radius, alpha_start):
        for i in range(max_radius, 0, -2):
            alpha = alpha_start * (1 - i / max_radius)
            glow_color = (*color, int(alpha))
            temp_surf = pygame.Surface((max_radius * 2, max_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (max_radius, max_radius), i)
            surface.blit(temp_surf, (center[0] - max_radius, center[1] - max_radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _draw_star(self, surface, color, center, radius, points=5):
        pts = []
        for i in range(points * 2):
            angle = i * math.pi / points - math.pi / 2
            r = radius if i % 2 == 0 else radius / 2.5
            pts.append((center[0] + r * math.cos(angle), center[1] + r * math.sin(angle)))
        pygame.gfxdraw.aapolygon(surface, pts, color)
        pygame.gfxdraw.filled_polygon(surface, pts, color)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lives": self.player_lives}
        
    def render(self):
        return self._get_observation()

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the game directly for testing
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Neon Onslaught")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print("\n" + "="*30)
    print("      NEON ONSLAUGHT")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not done:
        # --- Action Mapping for Human ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_SHIFT] else 0

        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting game...")
                obs, info = env.reset()

        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()