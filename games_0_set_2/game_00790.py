# Generated: 2025-08-27T14:48:17.592581
# Source Brief: brief_00790.md
# Brief Index: 790

        
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



# Use a dummy video driver to run headless, which is standard for Gym environments
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press space to fire in your last movement direction."
    )

    game_description = (
        "Pilot a robot in a top-down arena, blasting enemies for points. Aim for headshots to maximize your score!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAX_STEPS = 2000
    NUM_ENEMIES = 15

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_ARENA_BORDER = (50, 55, 70)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_HEALTH_FG = (100, 255, 100)
    COLOR_HEALTH_BG = (120, 40, 40)
    COLOR_WHITE = (255, 255, 255)

    # Player
    PLAYER_SPEED = 4
    PLAYER_SIZE = 18
    PLAYER_MAX_HEALTH = 100
    PLAYER_FIRE_COOLDOWN = 6 # frames

    # Enemy
    ENEMY_SPEED = 1.5
    ENEMY_SIZE = 14
    ENEMY_MAX_HEALTH = 20
    ENEMY_PATH_RADIUS = 50
    ENEMY_HEADSHOT_RADIUS = 5

    # Projectile
    PROJECTILE_SPEED = 10
    PROJECTILE_SIZE = 3

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

        # Initialize all state variables to None before reset
        self.player_pos = None
        self.player_health = None
        self.player_facing_direction = None
        self.player_fire_cooldown_timer = None
        self.player_hit_timer = None
        self.prev_space_held = None
        self.muzzle_flash_timer = None
        
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.game_over = False

        # self.reset() is called by gymnasium.make, no need to call it here.
        # The validation call that was here is also problematic for standard gym usage.
        # It's better to run validation separately. For the purpose of this fix,
        # we will remove it from __init__ to align with standard practices.


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_facing_direction = np.array([0, -1], dtype=np.float32) # Start facing up
        self.player_fire_cooldown_timer = 0
        self.player_hit_timer = 0
        self.prev_space_held = False
        self.muzzle_flash_timer = 0
        
        self.enemies = self._spawn_enemies()
        self.projectiles = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def _spawn_enemies(self):
        enemies = []
        for _ in range(self.NUM_ENEMIES):
            padding = self.ENEMY_PATH_RADIUS + 20
            center_x = self.np_random.uniform(padding, self.SCREEN_WIDTH - padding)
            center_y = self.np_random.uniform(padding, self.SCREEN_HEIGHT - padding)
            
            start_angle = self.np_random.uniform(0, 2 * math.pi)
            
            enemies.append({
                "center_pos": np.array([center_x, center_y], dtype=np.float32),
                "angle": start_angle,
                "pos": np.array([0,0], dtype=np.float32),
                "health": self.ENEMY_MAX_HEALTH,
                "hit_timer": 0
            })
        return enemies

    def step(self, action):
        reward = -0.01

        self._handle_input(action)
        self._update_timers()

        self._update_player(action)
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()
        
        collision_reward, player_damage = self._handle_collisions()
        reward += collision_reward
        self.player_health -= player_damage
        self.player_health = max(0, self.player_health)
        
        if player_damage > 0 and self.player_hit_timer == 0:
            self.player_hit_timer = 10

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if self.player_health <= 0:
                reward -= 10
            elif not self.enemies:
                reward += 50
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1 # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1 # Right
        
        if np.any(move_vec):
            self.player_facing_direction = move_vec

        if space_held and not self.prev_space_held and self.player_fire_cooldown_timer == 0:
            self._fire_projectile()
        
        self.prev_space_held = bool(space_held)

    def _fire_projectile(self):
        # SFX: Player shoot
        self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN
        self.muzzle_flash_timer = 2
        
        start_pos = self.player_pos + self.player_facing_direction * (self.PLAYER_SIZE / 2)
        
        self.projectiles.append({
            "pos": start_pos,
            "vel": self.player_facing_direction * self.PROJECTILE_SPEED
        })

    def _update_timers(self):
        if self.player_fire_cooldown_timer > 0: self.player_fire_cooldown_timer -= 1
        if self.player_hit_timer > 0: self.player_hit_timer -= 1
        if self.muzzle_flash_timer > 0: self.muzzle_flash_timer -= 1
        for enemy in self.enemies:
            if enemy["hit_timer"] > 0: enemy["hit_timer"] -= 1

    def _update_player(self, action):
        movement, _, _ = action
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1
        elif movement == 2: move_vec[1] = 1
        elif movement == 3: move_vec[0] = -1
        elif movement == 4: move_vec[0] = 1
        
        self.player_pos += move_vec * self.PLAYER_SPEED
        
        half_size = self.PLAYER_SIZE / 2
        self.player_pos[0] = np.clip(self.player_pos[0], half_size, self.SCREEN_WIDTH - half_size)
        self.player_pos[1] = np.clip(self.player_pos[1], half_size, self.SCREEN_HEIGHT - half_size)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['angle'] += self.ENEMY_SPEED * 0.05
            enemy['pos'][0] = enemy['center_pos'][0] + math.cos(enemy['angle']) * self.ENEMY_PATH_RADIUS
            enemy['pos'][1] = enemy['center_pos'][1] + math.sin(enemy['angle']) * self.ENEMY_PATH_RADIUS

    def _update_projectiles(self):
        self.projectiles[:] = [p for p in self.projectiles if 0 < p['pos'][0] < self.SCREEN_WIDTH and 0 < p['pos'][1] < self.SCREEN_HEIGHT]
        for proj in self.projectiles:
            proj['pos'] += proj['vel']

    def _update_particles(self):
        self.particles[:] = [p for p in self.particles if p['life'] > 1]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        player_damage = 0
        
        projectiles_to_keep = []
        for proj in self.projectiles:
            hit_enemy = False
            for enemy in self.enemies:
                dist = math.hypot(proj['pos'][0] - enemy['pos'][0], proj['pos'][1] - enemy['pos'][1])
                if dist < self.ENEMY_SIZE / 2:
                    hit_enemy = True
                    enemy['health'] -= 10
                    enemy['hit_timer'] = 5
                    reward += 1 # Reward for hitting
                    # SFX: Enemy hit
                    
                    if dist < self.ENEMY_HEADSHOT_RADIUS:
                        reward += 2
                        self.score += 2
                    
                    self.score += 1
                    break
            
            if not hit_enemy:
                projectiles_to_keep.append(proj)
        self.projectiles = projectiles_to_keep
        
        enemies_to_keep = []
        for enemy in self.enemies:
            if enemy['health'] > 0:
                enemies_to_keep.append(enemy)
            else:
                # SFX: Enemy explosion
                reward += 5
                self.score += 5
                self._spawn_explosion(enemy['pos'])
        self.enemies = enemies_to_keep
        
        # FIX: pygame.Rect requires integer coordinates.
        player_rect = pygame.Rect(int(self.player_pos[0] - self.PLAYER_SIZE/2), int(self.player_pos[1] - self.PLAYER_SIZE/2), self.PLAYER_SIZE, self.PLAYER_SIZE)
        for enemy in self.enemies:
            dist = math.hypot(player_rect.centerx - enemy['pos'][0], player_rect.centery - enemy['pos'][1])
            if dist < (self.PLAYER_SIZE / 2 + self.ENEMY_SIZE / 2):
                player_damage += 1
                # SFX: Player damage
        
        return reward, player_damage

    def _spawn_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            life = self.np_random.integers(15, 30)
            color_val = self.np_random.uniform(150, 255)
            color = (255, int(color_val), 0)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'color': color})

    def _check_termination(self):
        if self.player_health <= 0: return True
        if not self.enemies: return True
        # Truncation is handled separately now in Gymnasium
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_ARENA_BORDER, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 5)
        self._render_particles()
        self._render_enemies()
        self._render_player()
        self._render_projectiles()

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30.0))
            temp_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], alpha), (2, 2), 2)
            self.screen.blit(temp_surf, (int(p['pos'][0] - 2), int(p['pos'][1] - 2)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = (int(enemy['pos'][0]), int(enemy['pos'][1]))
            radius = int(self.ENEMY_SIZE / 2)
            color = self.COLOR_WHITE if enemy['hit_timer'] > 0 else self.COLOR_ENEMY
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], radius, color)
            
            if enemy['hit_timer'] > 0:
                 pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_HEADSHOT_RADIUS, self.COLOR_WHITE)

    def _render_player(self):
        pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
        half_size = int(self.PLAYER_SIZE / 2)
        player_rect = pygame.Rect(pos_int[0] - half_size, pos_int[1] - half_size, self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        color = self.COLOR_WHITE if self.player_hit_timer > 0 else self.COLOR_PLAYER
        
        glow_radius = int(self.PLAYER_SIZE * 1.2)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        alpha = max(0, 100 - self.player_hit_timer * 10)
        pygame.draw.circle(glow_surf, (*color, alpha), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (pos_int[0] - glow_radius, pos_int[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.draw.rect(self.screen, color, player_rect, border_radius=3)
        
        if self.muzzle_flash_timer > 0:
            flash_pos = self.player_pos + self.player_facing_direction * (self.PLAYER_SIZE / 2 + 5)
            flash_pos_int = (int(flash_pos[0]), int(flash_pos[1]))
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, flash_pos_int, 5)

    def _render_projectiles(self):
        for proj in self.projectiles:
            start_pos = (int(proj['pos'][0]), int(proj['pos'][1]))
            end_pos_vec = proj['pos'] - proj['vel'] * 0.5
            end_pos = (int(end_pos_vec[0]), int(end_pos_vec[1]))
            pygame.draw.aaline(self.screen, self.COLOR_PROJECTILE, start_pos, end_pos, 2)

    def _render_ui(self):
        health_bar_width = 200
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (15, 15, health_bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (15, 15, int(health_bar_width * health_ratio), 20))
        
        enemy_text = self.font_small.render(f"Enemies: {len(self.enemies)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(enemy_text, (15, 40))

        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 15, 15))
        self.screen.blit(score_text, score_rect)
        
        if self.game_over:
            end_text_str = "VICTORY!" if self.player_health > 0 and not self.enemies else "GAME OVER"
            end_text = self.font_large.render(end_text_str, True, self.COLOR_WHITE)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "enemies_left": len(self.enemies)
        }
        
    def close(self):
        pygame.quit()

def validate_implementation(env):
    """Helper function to validate a gym environment's implementation."""
    print("Beginning implementation validation...")
    assert env.action_space.shape == (3,)
    assert env.action_space.nvec.tolist() == [5, 2, 2]
    
    obs, info = env.reset()
    assert obs.shape == (env.SCREEN_HEIGHT, env.SCREEN_WIDTH, 3)
    assert obs.dtype == np.uint8
    assert isinstance(info, dict)
    
    test_action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(test_action)
    assert obs.shape == (env.SCREEN_HEIGHT, env.SCREEN_WIDTH, 3)
    assert isinstance(reward, (int, float))
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)
    
    env.reset()
    assert env.player_health == env.PLAYER_MAX_HEALTH
    assert len(env.enemies) == env.NUM_ENEMIES
    env.score = 100
    info = env._get_info()
    assert info["score"] == 100
    
    print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows direct execution for testing and visualization
    # To run with visualization, ensure a real display driver is available
    # by commenting out the `os.environ["SDL_VIDEODRIVER"] = "dummy"` line.
    
    # Example:
    # 1. Comment out the os.environ line at the top.
    # 2. Run the script. A window should appear.
    # 3. You won't be able to control it with the keyboard directly,
    #    as this is designed for an agent, but you can see random play.
    
    is_visual = False
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        print("\nRunning with visualization...")
        is_visual = True
    else:
        # Re-set it if it was not there, to be sure
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        print("\nRunning in headless mode...")
        is_visual = False

    env = GameEnv(render_mode="rgb_array")
    
    # Run validation
    validate_implementation(env)

    if is_visual:
        pygame.display.set_caption(env.game_description)
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    for episode in range(3):
        obs, info = env.reset()
        print(f"\n--- Episode {episode+1} ---")
        total_reward = 0
        
        for i in range(env.MAX_STEPS + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if is_visual:
                # The observation is (H, W, C), but pygame blit needs (W, H)
                # and the array is transposed from surfarray. So we need to
                # transpose it back.
                frame = np.transpose(obs, (1, 0, 2))
                pygame_surface = pygame.surfarray.make_surface(frame)
                display_screen.blit(pygame_surface, (0, 0))
                pygame.display.flip()
                env.clock.tick(30) # Limit to 30 FPS for viewing

            if terminated or truncated:
                if terminated:
                    print(f"Episode terminated after {i+1} steps. Final Info: {info}, Total Reward: {total_reward:.2f}")
                else: # truncated
                    print(f"Episode truncated at {i+1} steps. Final Info: {info}, Total Reward: {total_reward:.2f}")
                break
    
    env.close()