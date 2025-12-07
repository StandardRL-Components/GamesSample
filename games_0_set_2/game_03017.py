
# Generated: 2025-08-28T06:43:36.105764
# Source Brief: brief_03017.md
# Brief Index: 3017

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot. Survive the horde."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive a procedurally generated zombie horde for 60 seconds in a dark, isometric world."
    )

    # Frames auto-advance at 30fps.
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_PLAYER = (0, 200, 255)
    COLOR_ZOMBIE = (70, 170, 70)
    COLOR_BULLET = (255, 255, 255)
    COLOR_MUZZLE_FLASH = (255, 230, 150)
    COLOR_BLOOD = (136, 0, 0)
    COLOR_SHADOW = (15, 15, 25)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BAR_BG = (80, 0, 0)
    COLOR_HEALTH_BAR_FILL = (220, 0, 0)

    # Player settings
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4
    PLAYER_HEALTH_MAX = 100
    SHOOT_COOLDOWN = 6  # frames between shots

    # Zombie settings
    ZOMBIE_SIZE = 14
    ZOMBIE_SPEED = 1.2
    ZOMBIE_DAMAGE = 10
    ZOMBIE_SPAWN_RATE_INITIAL = 60  # frames
    ZOMBIE_SPAWN_RATE_FINAL = 30  # frames
    ZOMBIE_SPAWN_RAMP_UP_FRAMES = 900 # 30 seconds

    # Bullet settings
    BULLET_SPEED = 12
    BULLET_SIZE = 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_score = pygame.font.SysFont("monospace", 30, bold=True)

        # Game state variables are initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.player_facing_dir = None
        self.shoot_cooldown_timer = None
        self.zombies = None
        self.bullets = None
        self.particles = None
        self.zombie_spawn_timer = None
        self.zombie_spawn_rate = None
        self.steps = None
        self.score = None
        self.game_over = None
        
        # This will be set in reset()
        self.np_random = None
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_facing_dir = np.array([0, -1], dtype=np.float32) # Default up
        self.shoot_cooldown_timer = 0

        self.zombies = []
        self.bullets = []
        self.particles = []

        self.zombie_spawn_timer = self.ZOMBIE_SPAWN_RATE_INITIAL
        self.zombie_spawn_rate = self.ZOMBIE_SPAWN_RATE_INITIAL
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Reward for surviving a frame

        self._handle_input(action)
        self._update_player()
        
        killed_zombies_this_step, damage_taken_this_step = self._update_and_collide()
        reward += killed_zombies_this_step * 1.0
        self.score += killed_zombies_this_step
        self.player_health -= damage_taken_this_step

        self._update_spawner()
        self._update_particles()
        
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS

        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward = -100.0
            elif self.steps >= self.MAX_STEPS:
                reward = 50.0

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Movement and Facing Direction
        move_vector = np.array([0, 0], dtype=np.float32)
        if movement == 1:  # Up
            move_vector[1] -= 1
        elif movement == 2:  # Down
            move_vector[1] += 1
        elif movement == 3:  # Left
            move_vector[0] -= 1
        elif movement == 4:  # Right
            move_vector[0] += 1

        if np.linalg.norm(move_vector) > 0:
            self.player_pos += move_vector * self.PLAYER_SPEED
            self.player_facing_dir = move_vector
        
        # Shooting
        if self.shoot_cooldown_timer > 0:
            self.shoot_cooldown_timer -= 1
        
        if space_held and self.shoot_cooldown_timer == 0:
            self._shoot()

    def _update_player(self):
        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _shoot(self):
        # SFX: Player shoot sound
        self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
        
        bullet_pos = self.player_pos.copy()
        bullet_vel = self.player_facing_dir * self.BULLET_SPEED
        self.bullets.append({'pos': bullet_pos, 'vel': bullet_vel})
        
        # Muzzle flash particle
        flash_pos = self.player_pos + self.player_facing_dir * (self.PLAYER_SIZE + 5)
        self.particles.append({
            'pos': flash_pos, 'type': 'flash', 'lifetime': 2, 'radius': 12
        })

    def _update_and_collide(self):
        # Update bullets
        for b in self.bullets:
            b['pos'] += b['vel']
        
        # Update zombies
        for z in self.zombies:
            direction = self.player_pos - z['pos']
            dist = np.linalg.norm(direction)
            if dist > 1: # Avoid division by zero
                z['pos'] += (direction / dist) * self.ZOMBIE_SPEED
        
        # Collisions
        killed_zombies_this_step = 0
        damage_taken_this_step = 0
        
        remaining_bullets = []
        hit_zombie_indices = set()
        
        # Bullet-Zombie collisions
        for b_idx, b in enumerate(self.bullets):
            bullet_hit = False
            for z_idx, z in enumerate(self.zombies):
                if z_idx in hit_zombie_indices:
                    continue
                if np.linalg.norm(b['pos'] - z['pos']) < self.ZOMBIE_SIZE:
                    # SFX: Zombie hit sound
                    hit_zombie_indices.add(z_idx)
                    killed_zombies_this_step += 1
                    bullet_hit = True
                    self._create_blood_splatter(z['pos'])
                    break 
            if not bullet_hit:
                remaining_bullets.append(b)

        self.bullets = [b for b in remaining_bullets if 0 < b['pos'][0] < self.WIDTH and 0 < b['pos'][1] < self.HEIGHT]
        
        remaining_zombies = []
        # Player-Zombie collisions
        for z_idx, z in enumerate(self.zombies):
            if z_idx in hit_zombie_indices:
                continue
            if np.linalg.norm(self.player_pos - z['pos']) < self.PLAYER_SIZE + self.ZOMBIE_SIZE / 2:
                # SFX: Player damage sound
                damage_taken_this_step += self.ZOMBIE_DAMAGE
                self._create_blood_splatter(z['pos'], count=5) # Small splatter for contact
                # Zombie is consumed on contact
            else:
                remaining_zombies.append(z)
        
        self.zombies = remaining_zombies
        
        return killed_zombies_this_step, damage_taken_this_step

    def _create_blood_splatter(self, pos, count=15):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(), 'vel': vel, 'type': 'blood', 
                'lifetime': self.np_random.integers(15, 30), 'radius': self.np_random.uniform(1, 3)
            })

    def _update_spawner(self):
        # Linearly decrease spawn rate from initial to final over ramp-up time
        if self.steps < self.ZOMBIE_SPAWN_RAMP_UP_FRAMES:
            progress = self.steps / self.ZOMBIE_SPAWN_RAMP_UP_FRAMES
            self.zombie_spawn_rate = self.ZOMBIE_SPAWN_RATE_INITIAL - (self.ZOMBIE_SPAWN_RATE_INITIAL - self.ZOMBIE_SPAWN_RATE_FINAL) * progress
        else:
            self.zombie_spawn_rate = self.ZOMBIE_SPAWN_RATE_FINAL
        
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            self.zombie_spawn_timer = int(self.zombie_spawn_rate)
            
            # Spawn zombie at a random edge
            edge = self.np_random.integers(0, 4)
            if edge == 0: # Top
                pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE], dtype=np.float32)
            elif edge == 1: # Bottom
                pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE], dtype=np.float32)
            elif edge == 2: # Left
                pos = np.array([-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
            else: # Right
                pos = np.array([self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)], dtype=np.float32)
            
            self.zombies.append({'pos': pos})

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['lifetime'] -= 1
            if p['lifetime'] > 0:
                if p['type'] == 'blood':
                    p['pos'] += p['vel']
                    p['vel'] *= 0.9 # friction
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render shadows first
        self._render_shadow(self.player_pos, self.PLAYER_SIZE)
        for z in self.zombies:
            self._render_shadow(z['pos'], self.ZOMBIE_SIZE)

        # Render particles
        for p in self.particles:
            pos_int = p['pos'].astype(int)
            if p['type'] == 'blood':
                alpha = int(255 * (p['lifetime'] / 30))
                color = (*self.COLOR_BLOOD, alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color)
            elif p['type'] == 'flash':
                alpha = int(255 * (p['lifetime'] / 2))
                color = (*self.COLOR_MUZZLE_FLASH, alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color)
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], int(p['radius']), color)

        # Render zombies
        for z in self.zombies:
            pos_int = z['pos'].astype(int)
            pygame.draw.circle(self.screen, self.COLOR_ZOMBIE, pos_int, self.ZOMBIE_SIZE)

        # Render bullets
        for b in self.bullets:
            start_pos = b['pos'].astype(int)
            end_pos = (b['pos'] - b['vel'] * 0.5).astype(int) # Create a short tail
            pygame.draw.line(self.screen, self.COLOR_BULLET, start_pos, end_pos, self.BULLET_SIZE)

        # Render player
        player_pos_int = self.player_pos.astype(int)
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_pos_int, self.PLAYER_SIZE)
        # Facing direction indicator
        facing_end_pos = (self.player_pos + self.player_facing_dir * self.PLAYER_SIZE).astype(int)
        pygame.draw.line(self.screen, self.COLOR_BG, player_pos_int, facing_end_pos, 3)

    def _render_shadow(self, pos, radius):
        shadow_pos = (int(pos[0]), int(pos[1] + radius * 0.5))
        shadow_size = (int(radius * 2), int(radius))
        shadow_rect = pygame.Rect(0, 0, *shadow_size)
        shadow_rect.center = shadow_pos
        pygame.draw.ellipse(self.screen, self.COLOR_SHADOW, shadow_rect)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        bar_width = 200
        bar_height = 20
        bg_rect = pygame.Rect(10, 10, bar_width, bar_height)
        fill_rect = pygame.Rect(10, 10, int(bar_width * health_pct), bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FILL, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, bg_rect, 1)

        # Time Remaining
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {time_left:.1f}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))

        # Score
        score_text = f"{self.score:04d}"
        score_surf = self.font_score.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 25))
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "time_remaining": (self.MAX_STEPS - self.steps)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        # Reset first to initialize necessary components
        _ = self.reset()
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11', 'dummy' or 'windows' as appropriate

    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Create a window to display the game
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0
    
    print(env.user_guide)

    while not terminated:
        # Map keyboard keys to actions
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()