import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A side-view zombie survival shooter where the player must survive for a fixed duration.
    The player can move left/right, shoot zombies, and must reload their weapon.
    Zombies spawn in increasing waves, creating a tense, action-packed experience.
    Visuals are prioritized with particle effects for shooting, impacts, and deaths.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ← and → to move. Press Space to shoot and Shift to reload."
    )

    game_description = (
        "Survive for 5 minutes against a horde of zombies in this fast-paced side-scrolling shooter. "
        "Manage your ammo, aim carefully, and don't get overwhelmed!"
    )

    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GROUND_Y = 350
    FPS = 30 # For visual interpolation, not for step rate.

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_STAR = (200, 200, 220)
    COLOR_CITY = (40, 45, 60)
    COLOR_GROUND = (60, 55, 70)
    COLOR_PLAYER = (60, 220, 255)
    COLOR_PLAYER_GUN = (200, 200, 200)
    COLOR_ZOMBIE = (80, 140, 80)
    COLOR_BULLET = (255, 255, 100)
    COLOR_BLOOD = (200, 40, 40)
    COLOR_MUZZLE_FLASH = (255, 230, 150)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR = (40, 200, 80)
    COLOR_HEALTH_BAR_BG = (120, 40, 40)
    COLOR_AMMO_RELOAD = (255, 180, 20)

    # Game settings
    MAX_STEPS = 3000
    PLAYER_SPEED = 5.0
    PLAYER_HEALTH_MAX = 100
    PLAYER_AMMO_MAX = 30
    BULLET_SPEED = 20.0
    BULLET_DAMAGE = 10
    ZOMBIE_SPEED_MIN = 0.5
    ZOMBIE_SPEED_MAX = 1.5
    ZOMBIE_HEALTH_MAX = 30
    ZOMBIE_DAMAGE = 10
    ZOMBIE_INITIAL_SPAWN_INTERVAL = 40
    ZOMBIE_SPAWN_INTERVAL_MIN = 8
    ZOMBIE_DIFFICULTY_RAMP_STEPS = 600
    RELOAD_TIME = 60  # steps
    SHOOT_COOLDOWN = 6 # steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Pre-generate static background elements
        # This needs a temporary RNG for __init__ before reset() is called
        temp_seed = random.randint(0, 1_000_000)
        self.np_random = np.random.default_rng(seed=temp_seed)
        self.static_background = self._create_static_background()

        # Initialize state variables (these will be properly set in reset)
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_facing_right = None
        self.is_reloading = None
        self.reload_timer = None
        self.shoot_cooldown_timer = None
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.steps = None
        self.score = None
        self.game_over = None
        self.zombie_spawn_timer = None
        self.zombie_spawn_interval = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.screen_shake = 0
        self.player_vel_x = 0  # FIX: Initialize attribute
        
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.GROUND_Y)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_ammo = self.PLAYER_AMMO_MAX
        self.player_facing_right = True
        self.player_vel_x = 0 # FIX: Reset attribute
        
        self.is_reloading = False
        self.reload_timer = 0
        self.shoot_cooldown_timer = 0
        
        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.zombie_spawn_interval = self.ZOMBIE_INITIAL_SPAWN_INTERVAL
        self.zombie_spawn_timer = self.zombie_spawn_interval

        self.prev_space_held = False
        self.prev_shift_held = False
        self.screen_shake = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        
        self._handle_input(action)
        
        reward += self._update_player()
        reward += self._update_zombies()
        reward += self._update_bullets()
        self._update_particles()
        self._spawn_zombies()
        
        self.steps += 1
        
        terminated, terminal_reward = self._check_termination()
        reward += terminal_reward

        if terminated:
            self.game_over = True
        
        truncated = False # This environment does not truncate
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        self.player_vel_x = 0
        if movement == 3:  # Left
            self.player_vel_x = -self.PLAYER_SPEED
            self.player_facing_right = False
        elif movement == 4:  # Right
            self.player_vel_x = self.PLAYER_SPEED
            self.player_facing_right = True

        # Shooting (on key press)
        if space_held and not self.prev_space_held:
            if self.player_ammo > 0 and not self.is_reloading and self.shoot_cooldown_timer <= 0:
                self._shoot()
        self.prev_space_held = space_held

        # Reloading (on key press)
        if shift_held and not self.prev_shift_held:
            if not self.is_reloading and self.player_ammo < self.PLAYER_AMMO_MAX:
                self._reload()
        self.prev_shift_held = shift_held

    def _shoot(self):
        # sfx: gun_shot.wav
        self.player_ammo -= 1
        self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
        self.screen_shake = 5

        direction = 1 if self.player_facing_right else -1
        gun_tip_offset = pygame.Vector2(25 * direction, -22)
        bullet_start_pos = self.player_pos + gun_tip_offset
        
        bullet_vel = pygame.Vector2(self.BULLET_SPEED * direction, 0)
        self.bullets.append({"pos": bullet_start_pos, "vel": bullet_vel, "hit": False})

        # Muzzle flash
        self._create_particles(
            count=10,
            pos=bullet_start_pos,
            color=self.COLOR_MUZZLE_FLASH,
            speed_range=(2, 8),
            life_range=(2, 5),
            size_range=(4, 10),
            gravity=0.1
        )

    def _reload(self):
        # sfx: reload.wav
        self.is_reloading = True
        self.reload_timer = self.RELOAD_TIME

    def _update_player(self):
        # Update timers
        if self.shoot_cooldown_timer > 0:
            self.shoot_cooldown_timer -= 1
        if self.screen_shake > 0:
            self.screen_shake -= 1

        # Update position
        self.player_pos.x += self.player_vel_x
        self.player_pos.x = np.clip(self.player_pos.x, 15, self.WIDTH - 15)

        # Update reloading
        if self.is_reloading:
            self.reload_timer -= 1
            if self.reload_timer <= 0:
                self.is_reloading = False
                self.player_ammo = self.PLAYER_AMMO_MAX
                # sfx: reload_complete.wav
        return 0.0

    def _update_zombies(self):
        reward = 0.0
        for z in self.zombies[:]:
            # Movement
            if (self.player_pos - z['pos']).length() > 1: # Avoid division by zero
                direction = (self.player_pos - z['pos']).normalize()
                z['pos'] += direction * z['speed']
            
            # Animation timer
            z['anim_timer'] = (z['anim_timer'] + 1) % 20

            # Collision with player
            player_rect = pygame.Rect(self.player_pos.x - 10, self.player_pos.y - 40, 20, 40)
            zombie_rect = pygame.Rect(z['pos'].x - 10, z['pos'].y - 40, 20, 40)
            if player_rect.colliderect(zombie_rect):
                if z['attack_cooldown'] <= 0:
                    self.player_health -= self.ZOMBIE_DAMAGE
                    self.screen_shake = 10
                    z['attack_cooldown'] = self.FPS # 1 second cooldown
                    # sfx: player_hurt.wav
            
            if z['attack_cooldown'] > 0:
                z['attack_cooldown'] -= 1

            # Check health
            if z['health'] <= 0:
                self.zombies.remove(z)
                self.score += 1
                reward += 1.0 # Kill reward
                # sfx: zombie_death.wav
                self._create_particles(30, z['pos'] + pygame.Vector2(0, -20), self.COLOR_ZOMBIE, (1, 5), (10, 20), (2, 6), 0.2)
        return reward

    def _update_bullets(self):
        reward = 0.0
        for b in self.bullets[:]:
            b['pos'] += b['vel']

            # Off-screen check
            if not (0 < b['pos'].x < self.WIDTH):
                if not b['hit']:
                    reward -= 0.01  # Miss penalty
                self.bullets.remove(b)
                continue

            # Collision with zombies
            for z in self.zombies:
                zombie_rect = pygame.Rect(z['pos'].x - 12, z['pos'].y - 40, 24, 40)
                if zombie_rect.collidepoint(b['pos']):
                    z['health'] -= self.BULLET_DAMAGE
                    reward += 0.1  # Hit reward
                    b['hit'] = True
                    # sfx: bullet_impact.wav
                    self._create_particles(15, b['pos'], self.COLOR_BLOOD, (1, 4), (8, 15), (2, 4), 0.3)
                    if b in self.bullets:
                        self.bullets.remove(b)
                    break # Bullet hits one zombie and disappears
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'].y += p['gravity']
            p['life'] -= 1
            p['size'] -= 0.1
            if p['life'] <= 0 or p['size'] <= 0:
                self.particles.remove(p)

    def _spawn_zombies(self):
        self.zombie_spawn_timer -= 1
        if self.zombie_spawn_timer <= 0:
            side = self.np_random.choice([-1, 1])
            spawn_x = -20 if side == -1 else self.WIDTH + 20
            
            speed_multiplier = 1 + (self.steps / self.MAX_STEPS) * 1.5
            zombie_speed = self.np_random.uniform(self.ZOMBIE_SPEED_MIN, self.ZOMBIE_SPEED_MAX) * speed_multiplier

            self.zombies.append({
                "pos": pygame.Vector2(spawn_x, self.GROUND_Y),
                "health": self.ZOMBIE_HEALTH_MAX,
                "speed": zombie_speed,
                "attack_cooldown": 0,
                "anim_timer": self.np_random.integers(0, 20)
            })

            # Difficulty scaling
            ramp_up_factor = 1.0 - (self.steps / self.ZOMBIE_DIFFICULTY_RAMP_STEPS)
            ramp_up_factor = np.clip(ramp_up_factor, 0, 1)
            self.zombie_spawn_interval = self.ZOMBIE_SPAWN_INTERVAL_MIN + (self.ZOMBIE_INITIAL_SPAWN_INTERVAL - self.ZOMBIE_SPAWN_INTERVAL_MIN) * ramp_up_factor
            
            self.zombie_spawn_timer = self.zombie_spawn_interval

    def _check_termination(self):
        if self.player_health <= 0:
            return True, -100.0  # Died
        if self.steps >= self.MAX_STEPS:
            return True, 100.0   # Survived
        return False, 0.0

    def _get_observation(self):
        render_offset = pygame.Vector2(0, 0)
        if self.screen_shake > 0:
            render_offset.x = self.np_random.integers(-5, 6)
            render_offset.y = self.np_random.integers(-5, 6)

        self.screen.blit(self.static_background, render_offset)
        self._render_game(render_offset)
        self._render_ui() # UI is not affected by screen shake

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "ammo": self.player_ammo,
            "is_reloading": self.is_reloading
        }

    def _create_particles(self, count, pos, color, speed_range, life_range, size_range, gravity):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(speed_range[0], speed_range[1])
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                "life": self.np_random.integers(life_range[0], life_range[1]),
                "size": self.np_random.uniform(size_range[0], size_range[1]),
                "color": color,
                "gravity": gravity
            })

    def _create_static_background(self):
        bg_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        bg_surface.fill(self.COLOR_BG)
        # Stars
        for _ in range(100):
            x = self.np_random.integers(0, self.WIDTH)
            y = self.np_random.integers(0, self.GROUND_Y - 50)
            size = self.np_random.choice([1, 2])
            pygame.draw.rect(bg_surface, self.COLOR_STAR, (x, y, size, size))
        # City silhouette
        for i in range(15):
            x = self.np_random.integers(-20, self.WIDTH)
            w = self.np_random.integers(30, 80)
            h = self.np_random.integers(50, 200)
            y = self.GROUND_Y - h
            pygame.draw.rect(bg_surface, self.COLOR_CITY, (x, y, w, h))
        # Ground
        pygame.draw.rect(bg_surface, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        return bg_surface

    def _render_game(self, offset):
        # Particles
        for p in self.particles:
            pos = (int(p['pos'].x + offset.x), int(p['pos'].y + offset.y))
            size = max(0, int(p['size']))
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, p['color'])

        # Bullets
        for b in self.bullets:
            start = b['pos'] + offset
            end = b['pos'] - b['vel'].normalize() * 10 + offset
            pygame.draw.line(self.screen, self.COLOR_BULLET, (int(start.x), int(start.y)), (int(end.x), int(end.y)), 3)

        # Zombies
        for z in self.zombies:
            x, y = int(z['pos'].x + offset.x), int(z['pos'].y + offset.y)
            wobble = math.sin(z['anim_timer'] * math.pi / 10) * 2
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, (x - 10, y - 40 + wobble, 20, 40))
            pygame.draw.rect(self.screen, (50, 100, 50), (x - 7, y - 38 + wobble, 14, 30))

        # Player
        x, y = int(self.player_pos.x + offset.x), int(self.player_pos.y + offset.y)
        bob = math.sin(self.steps * 0.2) * 2 if self.player_vel_x != 0 else 0
        direction = 1 if self.player_facing_right else -1
        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (x - 12, y - 45 + bob, 24, 45))
        # Gun
        gun_y = y - 25 + bob
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GUN, (x - (5 if direction == 1 else 25), gun_y, 30, 8))
        
        # Reloading progress bar
        if self.is_reloading:
            progress = 1.0 - (self.reload_timer / self.RELOAD_TIME)
            bar_width = 40
            bar_height = 5
            bar_x = x - bar_width // 2
            bar_y = y - 60
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_AMMO_RELOAD, (bar_x, bar_y, int(bar_width * progress), bar_height))


    def _render_ui(self):
        # Health Bar
        health_ratio = np.clip(self.player_health / self.PLAYER_HEALTH_MAX, 0, 1)
        bar_width = 150
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_small.render(f"HP: {int(self.player_health)}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Ammo
        ammo_color = self.COLOR_TEXT if not self.is_reloading else self.COLOR_AMMO_RELOAD
        ammo_str = "RELOADING..." if self.is_reloading else f"AMMO: {self.player_ammo}/{self.PLAYER_AMMO_MAX}"
        ammo_text = self.font_small.render(ammo_str, True, ammo_color)
        self.screen.blit(ammo_text, (10, 35))

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        minutes = int(time_left // 60)
        seconds = int(time_left % 60)
        timer_text = self.font_small.render(f"TIME: {minutes:02}:{seconds:02}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        score_pos = (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - score_text.get_height() - 5)
        self.screen.blit(score_text, score_pos)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_health <= 0:
                end_text_str = "YOU DIED"
                end_color = self.COLOR_BLOOD
            else:
                end_text_str = "YOU SURVIVED!"
                end_color = self.COLOR_HEALTH_BAR
                
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_pos = (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2)
            self.screen.blit(end_text, text_pos)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
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
        assert self.zombie_spawn_timer > 0 # Assert zombie spawns after reset
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        # Test termination conditions
        self.player_health = 0
        _, _, term, _, _ = self.step(self.action_space.sample())
        assert term, "Termination on health=0 failed"
        self.reset()
        self.steps = self.MAX_STEPS
        _, _, term, _, _ = self.step(self.action_space.sample())
        assert term, "Termination on max_steps failed"
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Use a display for manual testing
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    
    while running:
        # --- Action polling ---
        keys = pygame.key.get_pressed()
        move_action = 0 # no-op
        if keys[pygame.K_LEFT]:
            move_action = 3
        elif keys[pygame.K_RIGHT]:
            move_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_SHIFT] else 0
        
        action = [move_action, space_action, shift_action]

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
        
        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to get it to the display screen
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS) # Control the frame rate

    env.close()