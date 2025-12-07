
# Generated: 2025-08-27T14:58:38.798976
# Source Brief: brief_00850.md
# Brief Index: 850

        
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
        "Controls: ↑ to jump, ←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a jumping, shooting robot through a side-scrolling obstacle course to reach the exit within the time limit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 10, 40)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_COCKPIT = (255, 255, 255)
    COLOR_PLAYER_GUN = (180, 180, 180)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PLAYER_PROJ = (100, 200, 255)
    COLOR_ENEMY_PROJ = (255, 150, 50)
    COLOR_OBSTACLE = (80, 80, 100)
    COLOR_EXIT = (200, 50, 255)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_PROGRESS_BAR_BG = (50, 50, 80)
    COLOR_PROGRESS_BAR_FG = (100, 200, 255)

    # Physics
    FPS = 30
    GRAVITY = 0.8
    PLAYER_SPEED = 5
    PLAYER_JUMP_STRENGTH = -15
    PROJECTILE_SPEED = 15

    # Game Rules
    MAX_STEPS = 120 * FPS  # 120 seconds
    INITIAL_LIVES = 3
    LEVEL_WIDTH_SCREENS = 15
    LEVEL_WIDTH = SCREEN_WIDTH * LEVEL_WIDTH_SCREENS
    FLOOR_HEIGHT = SCREEN_HEIGHT - 40
    PLAYER_SHOOT_COOLDOWN = 8  # frames
    ENEMY_SHOOT_COOLDOWN_RANGE = (45, 90)
    PLAYER_INVULNERABILITY_FRAMES = 60

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_big = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.player_rect = None
        self.player_lives = 0
        self.on_ground = False
        self.player_last_shot = 0
        self.player_invulnerable_timer = 0
        self.last_player_world_x = 0

        self.world_scroll_x = 0
        self.exit_pos_x = self.LEVEL_WIDTH - 150

        self.background_stars = []
        self.obstacles = []
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self.enemy_spawn_rate = 0.015
        self.enemy_projectile_speed = 4.0

        # Initialize state variables
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = pygame.Vector2(100, self.FLOOR_HEIGHT)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 30, 40)
        self.player_lives = self.INITIAL_LIVES
        self.on_ground = True
        self.player_last_shot = 0
        self.player_invulnerable_timer = 0
        self.last_player_world_x = self.player_pos.x
        
        self.world_scroll_x = 0
        
        self.enemies.clear()
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        
        self._generate_level()
        
        self.enemy_spawn_rate = 0.015
        self.enemy_projectile_speed = 4.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        self._handle_input(action)
        reward += self._update_player()
        self._update_enemies()
        self._update_projectiles()
        reward += self._handle_collisions()
        self._update_particles()
        self._update_world_state()

        self.steps += 1
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if self.player_pos.x >= self.exit_pos_x:
                # Goal-oriented reward
                time_bonus = (self.MAX_STEPS - self.steps) / self.MAX_STEPS
                win_reward = 50 + (50 * time_bonus)
                reward += win_reward
                self.score += win_reward

        # Continuous reward for progress
        progress = self.player_pos.x - self.last_player_world_x
        reward += progress * 0.1
        self.score += progress * 0.1
        self.last_player_world_x = self.player_pos.x
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_level(self):
        self.obstacles.clear()
        self.background_stars.clear()

        # Generate background stars
        for _ in range(200):
            self.background_stars.append({
                "pos": (self.np_random.integers(0, self.LEVEL_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)),
                "depth": self.np_random.uniform(0.1, 0.6)
            })

        # Generate obstacles
        current_x = 400
        while current_x < self.LEVEL_WIDTH - self.SCREEN_WIDTH:
            gap_size = self.np_random.integers(120, 300)
            current_x += gap_size
            
            obstacle_type = self.np_random.choice(['pit', 'wall', 'platform'])
            
            if obstacle_type == 'pit':
                # Pits are just gaps, so we just advance x
                current_x += self.np_random.integers(80, 150)
            elif obstacle_type == 'wall':
                height = self.np_random.integers(50, 150)
                self.obstacles.append(pygame.Rect(current_x, self.FLOOR_HEIGHT - height, 60, height))
                current_x += 60
            elif obstacle_type == 'platform':
                width = self.np_random.integers(100, 250)
                height = self.np_random.integers(100, 200)
                self.obstacles.append(pygame.Rect(current_x, self.FLOOR_HEIGHT - height, width, 20))
                current_x += width

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0

        # Jumping
        if movement == 1 and self.on_ground:
            self.player_vel.y = self.PLAYER_JUMP_STRENGTH
            self.on_ground = False
            # sfx: player_jump.wav
        
        # Shooting
        if space_held and (self.steps - self.player_last_shot > self.PLAYER_SHOOT_COOLDOWN):
            self.player_last_shot = self.steps
            proj_pos = self.player_pos + pygame.Vector2(self.player_rect.width, self.player_rect.height / 2 - 10)
            self.player_projectiles.append(proj_pos)
            self._create_particles(proj_pos, 5, self.COLOR_PLAYER_PROJ, 2, 4, 2) # Muzzle flash
            # sfx: player_shoot.wav

    def _update_player(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        
        # Update position
        self.player_pos += self.player_vel

        # Floor collision
        if self.player_pos.y > self.FLOOR_HEIGHT:
            if self.player_vel.y > 2: # Landing effect on hard fall
                self._create_particles(self.player_pos + pygame.Vector2(self.player_rect.width/2, self.player_rect.height), 5, (200,200,200), 1, 3, 2)
            self.player_pos.y = self.FLOOR_HEIGHT
            self.player_vel.y = 0
            self.on_ground = True

        # World boundaries
        self.player_pos.x = max(self.player_pos.x, 0)
        self.player_pos.x = min(self.player_pos.x, self.LEVEL_WIDTH - self.player_rect.width)

        self.player_rect.topleft = self.player_pos

        if self.player_invulnerable_timer > 0:
            self.player_invulnerable_timer -= 1
        
        return 0 # Reward is handled elsewhere

    def _update_enemies(self):
        # Spawn new enemies
        if self.np_random.random() < self.enemy_spawn_rate:
            spawn_x = self.world_scroll_x + self.SCREEN_WIDTH + 50
            if spawn_x < self.LEVEL_WIDTH - 100:
                self.enemies.append({
                    "rect": pygame.Rect(spawn_x, self.FLOOR_HEIGHT - 30, 30, 30),
                    "shoot_cooldown": self.np_random.integers(*self.ENEMY_SHOOT_COOLDOWN_RANGE)
                })

        # Update existing enemies
        for enemy in self.enemies:
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0:
                # sfx: enemy_shoot.wav
                proj_pos = pygame.Vector2(enemy["rect"].center) - pygame.Vector2(20, 0)
                self.enemy_projectiles.append({"pos": proj_pos, "vel": pygame.Vector2(-self.enemy_projectile_speed, 0)})
                enemy["shoot_cooldown"] = self.np_random.integers(*self.ENEMY_SHOOT_COOLDOWN_RANGE)

    def _update_projectiles(self):
        self.player_projectiles = [p + pygame.Vector2(self.PROJECTILE_SPEED, 0) for p in self.player_projectiles if p.x < self.world_scroll_x + self.SCREEN_WIDTH + 20]
        
        for proj in self.enemy_projectiles:
            proj["pos"] += proj["vel"]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p["pos"].x > self.world_scroll_x - 20]

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs enemies
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj.x, proj.y, 10, 4)
            for enemy in self.enemies[:]:
                if enemy["rect"].colliderect(proj_rect):
                    # sfx: explosion.wav
                    self._create_particles(enemy["rect"].center, 20, self.COLOR_ENEMY, 2, 8, 5)
                    self.enemies.remove(enemy)
                    self.player_projectiles.remove(proj)
                    reward += 5
                    self.score += 5
                    break
        
        if self.player_invulnerable_timer > 0:
            return reward

        # Player vs enemy projectiles, enemies, and obstacles
        damage_taken = False
        for proj in self.enemy_projectiles[:]:
            if self.player_rect.collidepoint(proj["pos"]):
                self.enemy_projectiles.remove(proj)
                damage_taken = True
                break
        
        if not damage_taken:
            for enemy in self.enemies:
                if self.player_rect.colliderect(enemy["rect"]):
                    damage_taken = True
                    break

        if not damage_taken:
            for obstacle in self.obstacles:
                if self.player_rect.colliderect(obstacle):
                    # Collision response: prevent movement into obstacle
                    # Check horizontal collision
                    if self.player_vel.x > 0 and self.player_rect.right > obstacle.left:
                        self.player_rect.right = obstacle.left
                    elif self.player_vel.x < 0 and self.player_rect.left < obstacle.right:
                        self.player_rect.left = obstacle.right
                    self.player_pos.x = self.player_rect.x

                    # Check vertical collision
                    if self.player_vel.y > 0 and self.player_rect.bottom > obstacle.top:
                        self.player_rect.bottom = obstacle.top
                        self.player_pos.y = self.player_rect.y
                        self.player_vel.y = 0
                        self.on_ground = True
                    elif self.player_vel.y < 0 and self.player_rect.top < obstacle.bottom:
                        self.player_rect.top = obstacle.bottom
                        self.player_pos.y = self.player_rect.y
                        self.player_vel.y = 0

        if damage_taken:
            # sfx: player_damage.wav
            self.player_lives -= 1
            self.player_invulnerable_timer = self.PLAYER_INVULNERABILITY_FRAMES
            reward -= 10 # Higher penalty for taking damage
            self.score -= 10
            self._create_particles(self.player_rect.center, 15, self.COLOR_PLAYER, 1, 5, 4)

        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _update_world_state(self):
        # Update camera scroll
        target_scroll = self.player_pos.x - self.SCREEN_WIDTH / 4
        self.world_scroll_x = max(0, min(self.LEVEL_WIDTH - self.SCREEN_WIDTH, target_scroll))

        # Difficulty scaling
        if self.steps > 0 and self.steps % 50 == 0:
            self.enemy_spawn_rate = min(0.05, self.enemy_spawn_rate * 1.01)
        if self.steps > 0 and self.steps % 100 == 0:
            self.enemy_projectile_speed = min(8.0, self.enemy_projectile_speed + 0.05)

    def _check_termination(self):
        return (
            self.player_lives <= 0
            or self.steps >= self.MAX_STEPS
            or self.player_pos.x >= self.exit_pos_x
        )

    def _create_particles(self, pos, count, color, min_speed, max_speed, life):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(min_speed, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(life, life * 2),
                "color": color,
                "size": self.np_random.uniform(2, 5)
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        scroll_x = int(self.world_scroll_x)

        # Background stars (parallax)
        for star in self.background_stars:
            x = (star["pos"][0] - scroll_x * star["depth"]) % self.SCREEN_WIDTH
            y = star["pos"][1]
            size = int(star["depth"] * 2)
            color_val = int(star["depth"] * 150) + 50
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (x, y, size, size))

        # Floor
        pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (0, self.FLOOR_HEIGHT + self.player_rect.height, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.FLOOR_HEIGHT))

        # Exit portal
        exit_screen_x = self.exit_pos_x - scroll_x
        if exit_screen_x < self.SCREEN_WIDTH:
            exit_rect = pygame.Rect(exit_screen_x, self.FLOOR_HEIGHT - 100, 20, 100 + self.player_rect.height)
            pulse = (math.sin(self.steps * 0.1) + 1) / 2
            color = (int(pulse * 100 + 155), 50, int(pulse * 100 + 155))
            pygame.draw.rect(self.screen, color, exit_rect)
            pygame.gfxdraw.rectangle(self.screen, exit_rect, (255,255,255))

        # Obstacles
        for obs in self.obstacles:
            if obs.right > scroll_x and obs.left < scroll_x + self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs.move(-scroll_x, 0))

        # Enemies
        for enemy in self.enemies:
            if enemy["rect"].right > scroll_x and enemy["rect"].left < scroll_x + self.SCREEN_WIDTH:
                pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy["rect"].move(-scroll_x, 0), border_radius=3)

        # Player
        if self.player_invulnerable_timer == 0 or self.steps % 10 < 5:
            player_screen_pos = self.player_pos - pygame.Vector2(scroll_x, 0)
            
            # Squash and stretch
            squash = max(0, -self.player_vel.y * 0.5) if not self.on_ground else 0
            stretch = max(0, self.player_vel.y * 0.3) if not self.on_ground else 0
            
            w = self.player_rect.width + squash
            h = self.player_rect.height + stretch
            x = player_screen_pos.x - squash/2
            y = player_screen_pos.y - stretch
            
            body_rect = pygame.Rect(x, y, w, h)
            gun_rect = pygame.Rect(x + w, y + h/2 - 10, 10, 6)
            cockpit_rect = pygame.Rect(x + w/4, y + h/4, w/2, h/4)

            pygame.draw.rect(self.screen, self.COLOR_PLAYER, body_rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_GUN, gun_rect, border_radius=2)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_COCKPIT, cockpit_rect, border_radius=2)

        # Projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (proj.x - scroll_x, proj.y, 10, 4), border_radius=2)
        for proj in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj["pos"].x - scroll_x), int(proj["pos"].y), 4, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, int(proj["pos"].x - scroll_x), int(proj["pos"].y), 4, self.COLOR_ENEMY_PROJ)

        # Particles
        for p in self.particles:
            size = p["size"] * (p["life"] / 10)
            if size > 0:
                pygame.draw.circle(self.screen, p["color"], p["pos"] - pygame.Vector2(scroll_x, 0), int(size))
    
    def _render_ui(self):
        # Lives
        for i in range(self.player_lives):
            pygame.gfxdraw.filled_circle(self.screen, 25 + i * 30, 25, 10, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, 25 + i * 30, 25, 10, self.COLOR_ENEMY)

        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) // self.FPS)
        time_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 15, 15))

        # Progress bar
        progress = self.player_pos.x / self.LEVEL_WIDTH
        bar_width = self.SCREEN_WIDTH / 2
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (bar_x, 15, bar_width, 20), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_FG, (bar_x, 15, bar_width * progress, 20), border_radius=4)

        if self.game_over:
            if self.player_lives <= 0:
                msg = "GAME OVER"
            elif self.steps >= self.MAX_STEPS:
                msg = "TIME UP"
            else: # Reached exit
                msg = "LEVEL COMPLETE"
            
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0,0,0,150))
            self.screen.blit(overlay, (0,0))
            
            end_text = self.font_big.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "time_remaining_steps": self.MAX_STEPS - self.steps,
            "distance_to_exit": max(0, self.exit_pos_x - self.player_pos.x)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    game_over_screen_timer = 0
    
    # Create a display for human play
    pygame.display.set_caption("GameEnv Test")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                game_over_screen_timer = 0

        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Game Over Logic ---
        if terminated:
            if game_over_screen_timer == 0:
                print(f"Episode finished. Final Info: {info}")
            game_over_screen_timer += 1
            if game_over_screen_timer > env.FPS * 3: # Wait 3 seconds
                print("Resetting environment.")
                obs, info = env.reset()
                game_over_screen_timer = 0

        # --- Frame Rate ---
        env.clock.tick(env.FPS)
        
    env.close()