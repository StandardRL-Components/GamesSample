
# Generated: 2025-08-27T16:31:13.896237
# Source Brief: brief_01250.md
# Brief Index: 1250

        
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
        "Controls: Arrow keys to move. Hold Space to shoot. Hold Shift to dash."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of zombies for 60 seconds in a top-down arena shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Game world
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (18, 18, 18)
    COLOR_ARENA = (40, 40, 40)
    COLOR_PLAYER = (0, 255, 127) # Bright Spring Green
    COLOR_ZOMBIE = (255, 69, 0) # Red-Orange
    COLOR_BULLET = (255, 255, 255)
    COLOR_MUZZLE_FLASH = (255, 223, 0)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR_BG = (139, 0, 0) # Dark Red
    COLOR_HEALTH_BAR_FG = (0, 128, 0) # Green

    # Player
    PLAYER_SIZE = 10
    PLAYER_SPEED = 4.0
    PLAYER_HEALTH = 100
    PLAYER_INVINCIBILITY_FRAMES = 30
    
    # Dash
    DASH_SPEED_MULTIPLIER = 3.0
    DASH_DURATION = 5 # frames
    DASH_COOLDOWN = 60 # frames
    
    # Shooting
    BULLET_SPEED = 10.0
    BULLET_SIZE = 3
    SHOOT_COOLDOWN = 6 # frames

    # Zombies
    ZOMBIE_SIZE = 12
    ZOMBIE_HEALTH = 10
    ZOMBIE_DAMAGE = 5
    
    # Difficulty Scaling
    INITIAL_ZOMBIE_SPEED = 1.0
    MAX_ZOMBIE_SPEED = 4.0
    ZOMBIE_SPEED_INCREASE_INTERVAL = 100 # steps
    ZOMBIE_SPEED_INCREASE_AMOUNT = 0.01

    INITIAL_SPAWN_RATE = 25 # frames
    MIN_SPAWN_RATE = 10 # frames
    SPAWN_RATE_DECREASE_INTERVAL = 200 # steps
    SPAWN_RATE_DECREASE_AMOUNT = 1

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize state variables to None
        self.player_pos = None
        self.player_health = None
        self.player_facing = None
        self.player_invincibility_timer = None
        
        self.dash_cooldown = None
        self.dash_timer = None
        
        self.shoot_cooldown = None
        self.prev_space_held = None
        self.prev_shift_held = None
        
        self.zombies = None
        self.bullets = None
        self.particles = None
        
        self.zombie_spawn_timer = None
        self.current_spawn_rate = None
        self.current_zombie_speed = None
        
        self.steps = None
        self.score = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH
        self.player_facing = pygame.math.Vector2(0, -1) # Start facing up
        self.player_invincibility_timer = 0
        
        self.dash_cooldown = 0
        self.dash_timer = 0
        
        self.shoot_cooldown = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.zombie_spawn_timer = 0
        self.current_spawn_rate = self.INITIAL_SPAWN_RATE
        self.current_zombie_speed = self.INITIAL_ZOMBIE_SPEED
        
        self.steps = 0
        self.score = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0.1 # Survival reward
        
        self._handle_input(movement, space_held, shift_held)
        
        self._update_player()
        self._update_bullets()
        self._update_zombies()
        self._update_particles()
        
        kill_reward = self._handle_collisions()
        reward += kill_reward
        
        self._handle_spawning()
        self._update_difficulty()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.player_health <= 0:
                reward = -50.0 # Death penalty
            elif self.steps >= self.MAX_STEPS:
                reward += 50.0 # Survival bonus
                
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        # --- Movement and Aiming ---
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_facing = move_vec.copy()

        # --- Dashing ---
        is_dashing = self.dash_timer > 0
        speed = self.PLAYER_SPEED * (self.DASH_SPEED_MULTIPLIER if is_dashing else 1)
        self.player_pos += move_vec * speed
        
        if shift_held and not self.prev_shift_held and self.dash_cooldown == 0:
            self.dash_timer = self.DASH_DURATION
            self.dash_cooldown = self.DASH_COOLDOWN
            # Dash particle effect
            for _ in range(20):
                self.particles.append(self._create_particle(self.player_pos, self.COLOR_PLAYER, 1, 3, 15))

        # --- Shooting ---
        if space_held and self.shoot_cooldown == 0:
            # Fire a bullet
            bullet_pos = self.player_pos + self.player_facing * (self.PLAYER_SIZE + 5)
            bullet_vel = self.player_facing * self.BULLET_SPEED
            self.bullets.append({'pos': bullet_pos, 'vel': bullet_vel})
            self.shoot_cooldown = self.SHOOT_COOLDOWN
            # Muzzle flash effect
            flash_pos = self.player_pos + self.player_facing * (self.PLAYER_SIZE + 2)
            self.particles.append(self._create_particle(flash_pos, self.COLOR_MUZZLE_FLASH, 8, 12, 3, is_circle=True))
            # Sound placeholder: # pygame.mixer.Sound("shoot.wav").play()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _update_player(self):
        # Update timers
        if self.shoot_cooldown > 0: self.shoot_cooldown -= 1
        if self.dash_timer > 0: self.dash_timer -= 1
        if self.dash_cooldown > 0: self.dash_cooldown -= 1
        if self.player_invincibility_timer > 0: self.player_invincibility_timer -= 1

        # Clamp player position to stay within arena
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_bullets(self):
        # Move bullets and remove off-screen ones
        self.bullets = [b for b in self.bullets if 0 < b['pos'].x < self.WIDTH and 0 < b['pos'].y < self.HEIGHT]
        for bullet in self.bullets:
            bullet['pos'] += bullet['vel']

    def _update_zombies(self):
        for zombie in self.zombies:
            direction = (self.player_pos - zombie['pos']).normalize()
            zombie['pos'] += direction * self.current_zombie_speed

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1

    def _handle_collisions(self):
        kill_reward = 0

        # --- Bullet-Zombie Collisions ---
        bullets_to_remove = []
        for i, bullet in enumerate(self.bullets):
            for zombie in self.zombies:
                if bullet['pos'].distance_to(zombie['pos']) < self.ZOMBIE_SIZE:
                    if i not in bullets_to_remove:
                        bullets_to_remove.append(i)
                    zombie['health'] -= 10
                    zombie['hit_timer'] = 5 # Flash white for 5 frames
                    # Sound placeholder: # pygame.mixer.Sound("hit.wav").play()
                    for _ in range(5):
                        self.particles.append(self._create_particle(bullet['pos'], self.COLOR_BULLET, 1, 2, 10))
                    break
        
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]

        # --- Zombie-Player Collisions and Zombie Deaths ---
        zombies_to_remove = []
        for i, zombie in enumerate(self.zombies):
            # Check for death
            if zombie['health'] <= 0:
                if i not in zombies_to_remove:
                    zombies_to_remove.append(i)
                self.score += 10
                kill_reward += 1.0
                # Sound placeholder: # pygame.mixer.Sound("zombie_die.wav").play()
                for _ in range(30):
                    self.particles.append(self._create_particle(zombie['pos'], self.COLOR_ZOMBIE, 1, 4, 20))
                continue

            # Check for player collision
            if self.player_invincibility_timer == 0 and self.player_pos.distance_to(zombie['pos']) < self.PLAYER_SIZE + self.ZOMBIE_SIZE:
                self.player_health -= self.ZOMBIE_DAMAGE
                self.player_invincibility_timer = self.PLAYER_INVINCIBILITY_FRAMES
                # Sound placeholder: # pygame.mixer.Sound("player_hurt.wav").play()
                if i not in zombies_to_remove:
                    zombies_to_remove.append(i) # Zombie dies on contact

        self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_to_remove]
        return kill_reward

    def _handle_spawning(self):
        self.zombie_spawn_timer += 1
        if self.zombie_spawn_timer >= self.current_spawn_rate:
            self.zombie_spawn_timer = 0
            
            # Spawn on a random edge
            side = self.np_random.integers(4)
            if side == 0: # Top
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE)
            elif side == 1: # Bottom
                pos = pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE)
            elif side == 2: # Left
                pos = pygame.math.Vector2(-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.math.Vector2(self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            
            self.zombies.append({'pos': pos, 'health': self.ZOMBIE_HEALTH, 'hit_timer': 0})

    def _update_difficulty(self):
        # Increase zombie speed
        if self.steps > 0 and self.steps % self.ZOMBIE_SPEED_INCREASE_INTERVAL == 0:
            self.current_zombie_speed = min(self.MAX_ZOMBIE_SPEED, self.current_zombie_speed + self.ZOMBIE_SPEED_INCREASE_AMOUNT)

        # Decrease spawn rate (faster spawning)
        if self.steps > 0 and self.steps % self.SPAWN_RATE_DECREASE_INTERVAL == 0:
            self.current_spawn_rate = max(self.MIN_SPAWN_RATE, self.current_spawn_rate - self.SPAWN_RATE_DECREASE_AMOUNT)

    def _check_termination(self):
        if self.player_health <= 0 or self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_ARENA)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'].x), int(p['pos'].y))
                if p.get('is_circle', False):
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])
                    pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])
                else:
                    size = max(0, int(p['radius']))
                    rect = pygame.Rect(pos[0] - size // 2, pos[1] - size // 2, size, size)
                    pygame.draw.rect(self.screen, p['color'], rect)

        # Bullets
        for bullet in self.bullets:
            pos = (int(bullet['pos'].x), int(bullet['pos'].y))
            pygame.draw.circle(self.screen, self.COLOR_BULLET, pos, self.BULLET_SIZE)

        # Zombies
        for z in self.zombies:
            pos = (int(z['pos'].x), int(z['pos'].y))
            color = (255, 255, 255) if z.get('hit_timer', 0) > 0 else self.COLOR_ZOMBIE
            if z.get('hit_timer', 0) > 0: z['hit_timer'] -= 1
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ZOMBIE_SIZE, color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ZOMBIE_SIZE, color)

        # Player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Player color flash on invincibility
        is_invincible = self.player_invincibility_timer > 0
        if is_invincible and (self.player_invincibility_timer // 3) % 2 == 0:
             # Don't draw player if flashing
             pass
        else:
            # Draw dash cooldown indicator
            if self.dash_cooldown > 0:
                ratio = self.dash_cooldown / self.DASH_COOLDOWN
                pygame.gfxdraw.arc(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE + 3, 90, 90 + int(360 * ratio), (255, 255, 0, 100))

            # Draw player circle
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)

            # Draw aiming indicator
            aim_end_pos = self.player_pos + self.player_facing * (self.PLAYER_SIZE)
            pygame.draw.line(self.screen, self.COLOR_BG, player_pos_int, (int(aim_end_pos.x), int(aim_end_pos.y)), 2)

    def _render_ui(self):
        # Health Bar
        health_ratio = max(0, self.player_health / self.PLAYER_HEALTH)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(bar_width * health_ratio), bar_height))
        
        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.FPS
        time_text = f"TIME: {max(0, time_left):.1f}"
        time_surf = self.font_small.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (self.WIDTH - time_surf.get_width() - 10, 10))
        
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.WIDTH // 2 - score_surf.get_width() // 2, self.HEIGHT - score_surf.get_height() - 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS,
        }
        
    def _create_particle(self, pos, color, min_radius, max_radius, lifespan, is_circle=False):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 4)
        vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
        radius = self.np_random.uniform(min_radius, max_radius)
        return {'pos': pos.copy(), 'vel': vel, 'radius': radius, 'color': color, 'lifespan': lifespan, 'is_circle': is_circle}

    def close(self):
        pygame.quit()
        
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
        
        # Test assertions from brief
        self.reset()
        assert self.player_health <= self.PLAYER_HEALTH
        assert (self.MAX_STEPS - self.steps) / self.FPS <= self.GAME_DURATION_SECONDS
        assert self.score >= 0
        assert self.current_zombie_speed <= self.MAX_ZOMBIE_SPEED
        assert self.current_spawn_rate >= self.MIN_SPAWN_RATE

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
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
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
            # Wait for a moment before restarting
            pygame.time.wait(2000)

        clock.tick(GameEnv.FPS)
        
    env.close()