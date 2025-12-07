import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing strings ---
    user_guide = (
        "Controls: Arrow keys to move and aim. Press space to fire. Press shift to dash."
    )
    game_description = (
        "Survive for 5 minutes against waves of zombies in a top-down arena shooter."
    )
    
    # --- Game Configuration ---
    auto_advance = True
    
    # Screen dimensions
    WIDTH, HEIGHT = 640, 400
    
    # Game constants
    FPS = 30
    WIN_TIME_SECONDS = 300
    MAX_STEPS = WIN_TIME_SECONDS * FPS + 10 # 5 minutes * 30fps + buffer
    WAVE_INTERVAL_SECONDS = 60
    
    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_WALL = (100, 100, 120)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 80, 80, 50)
    COLOR_ZOMBIE = (80, 200, 80)
    COLOR_ZOMBIE_GLOW = (80, 200, 80, 40)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_PROJECTILE_GLOW = (255, 255, 100, 100)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_BAR = (220, 50, 50)
    COLOR_TIMER_BAR = (50, 150, 220)
    COLOR_BAR_BG = (60, 60, 70)

    # Player settings
    PLAYER_SIZE = 10
    PLAYER_SPEED = 3.0
    PLAYER_HEALTH_MAX = 100
    PLAYER_DASH_SPEED = 12.0
    PLAYER_DASH_DURATION = 5 # frames
    PLAYER_DASH_COOLDOWN = 60 # frames
    PLAYER_HIT_INVULNERABILITY = 30 # frames

    # Weapon settings
    PROJECTILE_SIZE = 3
    PROJECTILE_SPEED = 10.0
    FIRE_COOLDOWN = 8 # frames

    # Zombie settings
    ZOMBIE_SIZE = 12
    ZOMBIE_HEALTH_MAX = 3
    ZOMBIE_BASE_SPEED = 0.8
    ZOMBIE_CONTACT_DAMAGE = 10
    ZOMBIE_SPAWN_PER_WAVE = 5
    ZOMBIE_SPEED_INCREASE_PER_WAVE = 0.2
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # State variables will be initialized in reset()
        self.player_pos = None
        self.player_health = None
        self.player_aim_angle = None
        self.player_last_move_vec = None
        self.player_dash_timer = None
        self.player_dash_cooldown = None
        self.player_invulnerability_timer = None
        
        self.zombies = None
        self.projectiles = None
        self.particles = None
        
        self.steps = None
        self.score = None
        self.game_over = None
        self.win_condition = None
        
        self.game_timer_steps = None
        self.wave_timer_steps = None
        self.current_wave = None
        
        self.fire_cooldown_timer = None
        self.last_space_held = None
        self.last_shift_held = None
        
        self.screen_shake = 0
        
        self.current_move_vec = pygame.math.Vector2(0, 0)

        # Initialize state (required for validation)
        # self.reset() is not called here to avoid double-initialization
        # but we need a seedable RNG for the validation call
        super().reset(seed=0)
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_aim_angle = 0.0
        self.player_last_move_vec = pygame.math.Vector2(1, 0)
        self.player_dash_timer = 0
        self.player_dash_cooldown = 0
        self.player_invulnerability_timer = 0
        
        # Game objects
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        # Game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        # Timers and wave management
        self.game_timer_steps = 0
        self.wave_timer_steps = self.WAVE_INTERVAL_SECONDS * self.FPS - 10 # First wave starts early
        self.current_wave = 0
        
        # Cooldowns and input tracking
        self.fire_cooldown_timer = 0
        self.last_space_held = 0
        self.last_shift_held = 0
        
        # Visuals
        self.screen_shake = 0

        # Initial zombie spawn
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            # If the game is over, do nothing but return the final state
            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info

        # --- Update Timers ---
        self.steps += 1
        self.game_timer_steps += 1
        self.wave_timer_steps += 1
        
        # --- Handle Input and Player Movement ---
        self._handle_input(action)
        self._update_player()
        
        # --- Update Game Objects ---
        self._update_projectiles()
        self._update_zombies()
        self._update_particles()
        
        # --- Handle Collisions and Game Logic ---
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        # --- Spawn New Wave ---
        if self.wave_timer_steps >= self.WAVE_INTERVAL_SECONDS * self.FPS:
            self._spawn_wave()
        
        # --- Per-step reward ---
        reward -= 0.001 # Small penalty for time passing to encourage action
        
        # --- Check Termination Conditions ---
        if self.player_health <= 0:
            self.game_over = True
            terminated = True
            reward -= 100 # Large penalty for dying
            # sfx: player_death
        
        if self.game_timer_steps >= self.WIN_TIME_SECONDS * self.FPS:
            self.game_over = True
            self.win_condition = True
            terminated = True
            reward += 100 # Large reward for winning
            # sfx: game_win

        if self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False, # truncated
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action
        
        # --- Movement and Aiming ---
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y -= 1 # Up
        elif movement == 2: move_vec.y += 1 # Down
        elif movement == 3: move_vec.x -= 1 # Left
        elif movement == 4: move_vec.x += 1 # Right
        
        if move_vec.length_squared() > 0:
            move_vec.normalize_ip()
            self.player_last_move_vec = pygame.math.Vector2(move_vec)
            self.player_aim_angle = math.atan2(-move_vec.y, move_vec.x)
        self.current_move_vec = move_vec

        # --- Dashing ---
        if shift_held and not self.last_shift_held and self.player_dash_cooldown <= 0:
            self.player_dash_timer = self.PLAYER_DASH_DURATION
            self.player_dash_cooldown = self.PLAYER_DASH_COOLDOWN
            # sfx: dash
            for _ in range(15):
                self._create_particle(self.player_pos, self.COLOR_PLAYER, 10, 2, 4)

        # --- Firing ---
        if space_held and not self.last_space_held and self.fire_cooldown_timer <= 0:
            self._fire_projectile()
            self.fire_cooldown_timer = self.FIRE_COOLDOWN
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _fire_projectile(self):
        # sfx: shoot_laser
        start_pos = self.player_pos + pygame.math.Vector2(self.PLAYER_SIZE, 0).rotate_rad(self.player_aim_angle)
        velocity = pygame.math.Vector2(self.PROJECTILE_SPEED, 0).rotate_rad(self.player_aim_angle)
        self.projectiles.append({"pos": start_pos, "vel": velocity})
        # Muzzle flash
        self._create_particle(start_pos, self.COLOR_PROJECTILE, 5, 2, 8, is_circle=True)
        self.screen_shake = max(self.screen_shake, 2)


    def _update_player(self):
        # Cooldowns
        if self.fire_cooldown_timer > 0: self.fire_cooldown_timer -= 1
        if self.player_dash_cooldown > 0: self.player_dash_cooldown -= 1
        if self.player_invulnerability_timer > 0: self.player_invulnerability_timer -= 1

        # Movement
        current_speed = self.PLAYER_SPEED
        if self.player_dash_timer > 0:
            current_speed = self.PLAYER_DASH_SPEED
            self.player_dash_timer -= 1
            # Create dash trail
            if self.steps % 2 == 0:
                 self._create_particle(self.player_pos, self.COLOR_PLAYER, 8, 1, 2, alpha=100)
        
        self.player_pos += self.current_move_vec * current_speed

        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            if not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_zombies(self):
        current_speed = self.ZOMBIE_BASE_SPEED + self.current_wave * self.ZOMBIE_SPEED_INCREASE_PER_WAVE
        for z in self.zombies:
            if z["pos"].distance_to(self.player_pos) > 1e-6: # Avoid normalization of zero vector
                direction = (self.player_pos - z["pos"]).normalize()
                z["pos"] += direction * current_speed

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Projectile-Zombie collisions
        for p in self.projectiles[:]:
            for z in self.zombies[:]:
                if p["pos"].distance_to(z["pos"]) < self.ZOMBIE_SIZE + self.PROJECTILE_SIZE:
                    # sfx: zombie_hit
                    if p in self.projectiles: self.projectiles.remove(p)
                    z["health"] -= 1
                    reward += 0.1 # Reward for hitting
                    self.screen_shake = max(self.screen_shake, 3)
                    for _ in range(5):
                        self._create_particle(p["pos"], self.COLOR_PROJECTILE, 8, 1, 3)

                    if z["health"] <= 0:
                        # sfx: zombie_death
                        self.zombies.remove(z)
                        self.score += 1
                        reward += 1.0 # Reward for kill
                        self.screen_shake = max(self.screen_shake, 6)
                        for _ in range(20):
                            self._create_particle(z["pos"], self.COLOR_ZOMBIE, 15, 0.5, 5)
                    break # Projectile can only hit one zombie

        # Zombie-Player collisions
        if self.player_invulnerability_timer <= 0:
            for z in self.zombies:
                if self.player_pos.distance_to(z["pos"]) < self.PLAYER_SIZE + self.ZOMBIE_SIZE:
                    # sfx: player_hit
                    self.player_health -= self.ZOMBIE_CONTACT_DAMAGE
                    self.player_invulnerability_timer = self.PLAYER_HIT_INVULNERABILITY
                    self.screen_shake = max(self.screen_shake, 8)
                    break # Only take damage from one zombie per frame
        
        return reward

    def _spawn_wave(self):
        self.wave_timer_steps = 0
        self.current_wave += 1
        num_to_spawn = self.current_wave * self.ZOMBIE_SPAWN_PER_WAVE
        
        for _ in range(num_to_spawn):
            # Spawn on edges of the screen
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                x, y = self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE
            elif edge == 1: # Bottom
                x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE
            elif edge == 2: # Left
                x, y = -self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)
            else: # Right
                x, y = self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT)
            
            self.zombies.append({
                "pos": pygame.math.Vector2(x, y),
                "health": self.ZOMBIE_HEALTH_MAX
            })

    def _create_particle(self, pos, color, lifetime, min_speed, max_speed, alpha=255, is_circle=False):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(min_speed, max_speed)
        vel = pygame.math.Vector2(speed, 0).rotate_rad(angle)
        self.particles.append({
            "pos": pygame.math.Vector2(pos),
            "vel": vel,
            "life": lifetime,
            "color": color,
            "alpha": alpha,
            "is_circle": is_circle
        })

    def _get_observation(self):
        # --- Rendering ---
        self.screen.fill(self.COLOR_BG)
        
        # Screen shake offset
        shake_offset = pygame.math.Vector2(0,0)
        if self.screen_shake > 0:
            shake_offset.x = self.np_random.uniform(-self.screen_shake, self.screen_shake)
            shake_offset.y = self.np_random.uniform(-self.screen_shake, self.screen_shake)
            self.screen_shake *= 0.9 # Decay
            if self.screen_shake < 0.5: self.screen_shake = 0

        # Render all game elements
        self._render_game(shake_offset)
        
        # Render UI overlay (not affected by shake)
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, offset):
        # Draw arena walls
        wall_rect = pygame.Rect(5, 5, self.WIDTH - 10, self.HEIGHT - 10)
        pygame.draw.rect(self.screen, self.COLOR_WALL, wall_rect.move(offset.x, offset.y), 3)

        # Draw particles
        for p in self.particles:
            pos = p["pos"] + offset
            life_ratio = p["life"] / 10.0
            size = int(max(1, life_ratio * 5))
            color = (*p["color"], int(p["alpha"] * life_ratio))
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            if p["is_circle"]:
                pygame.draw.circle(temp_surf, color, (size, size), size)
            else:
                pygame.draw.rect(temp_surf, color, (0, 0, size, size))
            self.screen.blit(temp_surf, (int(pos.x - size), int(pos.y - size)), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw projectiles
        for p in self.projectiles:
            pos = p["pos"] + offset
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.PROJECTILE_SIZE + 3, self.COLOR_PROJECTILE_GLOW)
            # Core
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.PROJECTILE_SIZE, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.PROJECTILE_SIZE, self.COLOR_PROJECTILE)

        # Draw zombies
        for z in self.zombies:
            pos = z["pos"] + offset
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.ZOMBIE_SIZE + 4, self.COLOR_ZOMBIE_GLOW)
            # Core
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), self.ZOMBIE_SIZE, self.COLOR_ZOMBIE)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), self.ZOMBIE_SIZE, self.COLOR_ZOMBIE)

        # Draw player
        player_pos_int = (int(self.player_pos.x + offset.x), int(self.player_pos.y + offset.y))
        
        # Invulnerability flash
        if self.player_invulnerability_timer > 0 and self.steps % 4 < 2:
            pass # Don't draw player to make them flash
        else:
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE + 6, self.COLOR_PLAYER_GLOW)
            # Core
            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
            
            # Aiming indicator
            end_point = self.player_pos + pygame.math.Vector2(self.PLAYER_SIZE + 5, 0).rotate_rad(self.player_aim_angle) + offset
            pygame.draw.line(self.screen, self.COLOR_TEXT, player_pos_int, (int(end_point.x), int(end_point.y)), 2)

    def _render_ui(self):
        # Health bar
        health_ratio = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 10, 200, 15))
        if health_ratio > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(200 * health_ratio), 15))

        # Timer bar
        time_ratio = min(1.0, self.game_timer_steps / (self.WIN_TIME_SECONDS * self.FPS))
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (10, 30, 200, 10))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (10, 30, int(200 * time_ratio), 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 30))

        # Game Over / Win message
        if self.game_over:
            if self.win_condition:
                msg = "YOU SURVIVED"
                color = self.COLOR_TIMER_BAR
            else:
                msg = "YOU DIED"
                color = self.COLOR_HEALTH_BAR
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "wave": self.current_wave,
            "game_timer": self.game_timer_steps / self.FPS
        }
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # This needs a fully initialized env, so we call reset first.
        obs, info = self.reset(seed=42)
        
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
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
    # This block allows you to play the game directly
    # It requires a display, so it will not run in a headless environment
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11"
        import pygame
        
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
        pygame.display.set_caption("Zombie Survival")
        clock = pygame.time.Clock()
        
        running = True
        total_reward = 0
        
        # --- Action state ---
        movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
        space_held = 0
        shift_held = 0
        
        print(GameEnv.game_description)
        print(GameEnv.user_guide)

        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # --- Get Player Input ---
            keys = pygame.key.get_pressed()
            
            # Movement (only one direction at a time as per action space)
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            else:
                movement = 0
                
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # --- Render the Observation ---
            # Pygame uses a different coordinate system, so we need to transpose
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Check for Game Over ---
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                # Wait a bit before closing or resetting
                pygame.time.wait(3000)
                running = False # or reset the game: obs, info = env.reset()
                
            # --- Control Framerate ---
            clock.tick(GameEnv.FPS)
            
        env.close()

    except pygame.error as e:
        print(f"Could not run in interactive mode: {e}")
        print("This is expected in a headless environment. The environment class is still valid.")
        # Create an instance just to run validation in headless mode
        env = GameEnv()
        print("Headless validation check passed.")