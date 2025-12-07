
# Generated: 2025-08-27T22:00:47.346163
# Source Brief: brief_02982.md
# Brief Index: 2982

        
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
        "Controls: Arrow keys to move. Hold Space to shoot in your last direction of movement."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive waves of zombies in a top-down arena for 60 seconds. Keep moving and shooting to stay alive!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_BULLET = (255, 255, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR_BG = (80, 0, 0)
        self.COLOR_HEALTH_BAR_FG = (0, 200, 0)

        # Player
        self.PLAYER_SIZE = 16
        self.PLAYER_SPEED = 4.0
        self.PLAYER_MAX_HEALTH = 100
        
        # Weapon
        self.BULLET_SPEED = 10.0
        self.BULLET_SIZE = 4
        self.SHOOT_COOLDOWN = 6 # frames (5 shots/sec)
        
        # Zombie
        self.ZOMBIE_SIZE = 18
        self.ZOMBIE_SPEED = 1.2
        self.ZOMBIE_DAMAGE = 10
        self.INITIAL_ZOMBIE_SPAWN_INTERVAL = 2 * self.FPS # 2 seconds
        
        # Game
        self.GAME_DURATION_SECONDS = 60
        self.GAME_DURATION_FRAMES = self.GAME_DURATION_SECONDS * self.FPS
        
        # Rewards
        self.REWARD_PER_SECOND = 0.1
        self.REWARD_KILL = 1.0
        self.REWARD_WIN = 50.0
        self.REWARD_LOSE = -10.0
        
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
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_game_over = pygame.font.Font(pygame.font.get_default_font(), 48)
        except IOError:
            self.font_ui = pygame.font.SysFont("arial", 18)
            self.font_game_over = pygame.font.SysFont("arial", 48)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.last_move_dir = None
        self.zombies = []
        self.bullets = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_timer = 0
        self.zombie_spawn_timer = 0
        self.zombie_spawn_interval = 0
        self.shoot_cooldown_timer = 0
        self.muzzle_flash_timer = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.last_move_dir = pygame.Vector2(0, -1) # Default aim up
        
        self.zombies.clear()
        self.bullets.clear()
        self.particles.clear()
        
        self.steps = 0
        self.score = 0
        self.game_timer = self.GAME_DURATION_FRAMES
        self.zombie_spawn_interval = self.INITIAL_ZOMBIE_SPAWN_INTERVAL
        self.zombie_spawn_timer = self.zombie_spawn_interval
        self.shoot_cooldown_timer = 0
        self.muzzle_flash_timer = 0
        self.game_over = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            # If the game is over, do not advance the state.
            # Just return the final observation and info.
            return self._get_observation(), 0, True, False, self._get_info()

        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = self.REWARD_PER_SECOND / self.FPS
        
        self._handle_input(action)
        self._update_state()
        
        collision_reward = self._handle_collisions()
        reward += collision_reward
        
        terminated = (self.player_health <= 0) or (self.game_timer <= 0)
        
        if terminated:
            self.game_over = True
            if self.player_health > 0: # Win condition
                reward += self.REWARD_WIN
                self.score += 100 # Bonus points for winning
            else: # Lose condition
                reward += self.REWARD_LOSE

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.last_move_dir = move_vec.copy()
        
        self.player_pos += move_vec * self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE / 2, self.WIDTH - self.PLAYER_SIZE / 2)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE / 2, self.HEIGHT - self.PLAYER_SIZE / 2)

        if space_held and self.shoot_cooldown_timer == 0:
            self._fire_bullet()

    def _fire_bullet(self):
        # sfx: Laser shot
        self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
        bullet_pos = self.player_pos + self.last_move_dir * (self.PLAYER_SIZE / 2)
        self.bullets.append({"pos": bullet_pos, "vel": self.last_move_dir * self.BULLET_SPEED})
        self.muzzle_flash_timer = 3 # frames

    def _update_state(self):
        # Timers
        self.game_timer -= 1
        self.shoot_cooldown_timer = max(0, self.shoot_cooldown_timer - 1)
        self.zombie_spawn_timer -= 1
        self.muzzle_flash_timer = max(0, self.muzzle_flash_timer - 1)

        # Update entities
        self._update_bullets()
        self._update_zombies()
        self._update_particles()
        self._spawn_zombies()
        self._update_spawn_rate()

    def _update_bullets(self):
        for bullet in self.bullets[:]:
            bullet["pos"] += bullet["vel"]
            if not self.screen.get_rect().collidepoint(bullet["pos"]):
                self.bullets.remove(bullet)

    def _update_zombies(self):
        for zombie in self.zombies:
            direction = (self.player_pos - zombie["pos"]).normalize()
            zombie["pos"] += direction * self.ZOMBIE_SPEED
    
    def _update_particles(self):
        for particle in self.particles[:]:
            particle["pos"] += particle["vel"]
            particle["life"] -= 1
            if particle["life"] <= 0:
                self.particles.remove(particle)

    def _spawn_zombies(self):
        if self.zombie_spawn_timer <= 0:
            side = self.np_random.integers(0, 4)
            if side == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_SIZE)
            elif side == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_SIZE)
            elif side == 2: # Left
                pos = pygame.Vector2(-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.HEIGHT))
            
            self.zombies.append({"pos": pos, "rect": pygame.Rect(0,0,0,0)})
            self.zombie_spawn_timer = self.zombie_spawn_interval

    def _update_spawn_rate(self):
        # Increase spawn rate every 10 seconds
        elapsed_seconds = self.GAME_DURATION_SECONDS - (self.game_timer / self.FPS)
        # 0.01 increase per 10 seconds is equivalent to decreasing interval
        # This is a simplified approach to increasing difficulty over time
        if self.steps > 0 and self.steps % (10 * self.FPS) == 0:
             self.zombie_spawn_interval = max(15, self.zombie_spawn_interval * 0.9) # 15 frames min interval

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE / 2, self.player_pos.y - self.PLAYER_SIZE / 2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Update zombie rects
        for z in self.zombies:
            z["rect"] = pygame.Rect(z["pos"].x - self.ZOMBIE_SIZE/2, z["pos"].y - self.ZOMBIE_SIZE/2, self.ZOMBIE_SIZE, self.ZOMBIE_SIZE)

        # Bullets vs Zombies
        for bullet in self.bullets[:]:
            bullet_rect = pygame.Rect(bullet["pos"].x - self.BULLET_SIZE/2, bullet["pos"].y - self.BULLET_SIZE/2, self.BULLET_SIZE, self.BULLET_SIZE)
            for zombie in self.zombies[:]:
                if bullet_rect.colliderect(zombie["rect"]):
                    # sfx: Squish / Explosion
                    self.zombies.remove(zombie)
                    if bullet in self.bullets: self.bullets.remove(bullet)
                    self.score += 10
                    reward += self.REWARD_KILL
                    self._create_explosion(zombie["pos"], self.COLOR_ZOMBIE)
                    break
        
        # Player vs Zombies
        for zombie in self.zombies[:]:
            if player_rect.colliderect(zombie["rect"]):
                # sfx: Player hit
                self.zombies.remove(zombie)
                self.player_health -= self.ZOMBIE_DAMAGE
                self._create_explosion(self.player_pos, self.COLOR_PLAYER)

        return reward

    def _create_explosion(self, pos, color):
        for _ in range(self.np_random.integers(10, 20)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": pos.copy(), "vel": vel, "life": life, "color": color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = p["color"]
            size = max(1, int(p["life"] / 6))
            pygame.draw.rect(self.screen, color, (int(p["pos"].x), int(p["pos"].y), size, size))

        # Render zombies
        for z in self.zombies:
            pygame.draw.rect(self.screen, self.COLOR_ZOMBIE, z["rect"])
        
        # Render player glow
        glow_size = int(self.PLAYER_SIZE * 2.5)
        s = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(s, (int(self.player_pos.x - glow_size/2), int(self.player_pos.y - glow_size/2)))

        # Render player
        player_rect = pygame.Rect(0, 0, self.PLAYER_SIZE, self.PLAYER_SIZE)
        player_rect.center = (int(self.player_pos.x), int(self.player_pos.y))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Render bullets
        for b in self.bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET, (int(b["pos"].x - self.BULLET_SIZE/2), int(b["pos"].y - self.BULLET_SIZE/2), self.BULLET_SIZE, self.BULLET_SIZE))

        # Render muzzle flash
        if self.muzzle_flash_timer > 0:
            flash_pos = self.player_pos + self.last_move_dir * (self.PLAYER_SIZE / 2 + 5)
            flash_size = 10
            pygame.draw.circle(self.screen, (255, 255, 150), (int(flash_pos.x), int(flash_pos.y)), flash_size // 2)

    def _render_ui(self):
        # Health Bar
        health_bar_width = 150
        health_bar_height = 15
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, health_bar_width, health_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_FG, (10, 10, int(health_bar_width * health_pct), health_bar_height))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, 20))
        self.screen.blit(score_text, score_rect)

        # Timer
        time_left = max(0, int(self.game_timer / self.FPS))
        timer_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)
        
        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_health > 0:
                end_text_str = "YOU SURVIVED!"
                end_color = self.COLOR_PLAYER
            else:
                end_text_str = "GAME OVER"
                end_color = self.COLOR_ZOMBIE
                
            end_text = self.font_game_over.render(end_text_str, True, end_color)
            end_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "time_left": max(0, int(self.game_timer / self.FPS)),
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

if __name__ == "__main__":
    # This block allows you to run the file directly to play the game
    # It's useful for testing and debugging
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for display ---
    pygame.display.set_caption("Zombie Arena")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Display ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Wait 3 seconds
            obs, info = env.reset()
            total_reward = 0

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

    env.close()