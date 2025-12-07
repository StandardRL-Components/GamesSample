
# Generated: 2025-08-28T05:23:25.369959
# Source Brief: brief_05558.md
# Brief Index: 5558

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a top-down isometric zombie survival game.
    The player must survive for 5 minutes against hordes of zombies while managing health and ammo.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press space to fire your weapon in the direction you are facing."
    )

    game_description = (
        "Survive hordes of procedurally generated zombies in a 2D arena for 5 minutes by collecting supplies and strategically eliminating threats."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 3000
        self.INITIAL_ZOMBIES = 50
        self.PLAYER_SPEED = 3.0
        self.PLAYER_RADIUS = 12
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 100
        self.PLAYER_INITIAL_AMMO = 50
        self.PLAYER_HIT_COOLDOWN_STEPS = 20 # 2 seconds at 10 steps/sec
        self.ZOMBIE_RADIUS = 10
        self.INITIAL_ZOMBIE_SPEED = 1.0
        self.BULLET_RADIUS = 3
        self.BULLET_SPEED = 8.0
        self.SHOT_COOLDOWN_STEPS = 5 # 2 shots/sec
        self.CRATE_SPAWN_INTERVAL = 600
        self.DIFFICULTY_INTERVAL = 300
        self.CRATE_SIZE = 16

        # --- Colors ---
        self.COLOR_BG = (34, 34, 34)
        self.COLOR_ARENA_BORDER = (80, 80, 80)
        self.COLOR_PLAYER = (0, 200, 255)
        self.COLOR_PLAYER_HIT = (255, 50, 50)
        self.COLOR_ZOMBIE = (70, 140, 70)
        self.COLOR_BULLET = (255, 255, 255)
        self.COLOR_SHADOW = (0, 0, 0, 60)
        self.COLOR_CRATE_HEALTH = (0, 255, 100)
        self.COLOR_CRATE_AMMO = (100, 150, 255)
        self.COLOR_CRATE_BONUS = (255, 220, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_HEALTH = (0, 220, 120)
        self.COLOR_UI_DANGER = (200, 50, 50)
        self.COLOR_UI_BONUS = (255, 220, 0)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_timer = pygame.font.Font(None, 36)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_health = 0
        self.player_ammo = 0
        self.player_facing_direction = np.zeros(2, dtype=np.float32)
        self.player_hit_cooldown = 0
        self.zombies = []
        self.zombie_speed = 0.0
        self.bullets = []
        self.shot_cooldown = 0
        self.supply_crates = []
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_INITIAL_AMMO
        self.player_facing_direction = np.array([0, -1], dtype=np.float32) # Start facing up
        self.player_hit_cooldown = 0

        self.zombie_speed = self.INITIAL_ZOMBIE_SPEED
        self.zombies = []
        for _ in range(self.INITIAL_ZOMBIES):
            self._spawn_zombie()

        self.bullets = []
        self.shot_cooldown = 0
        self.supply_crates = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1 # Survival reward per step

        # --- Update Cooldowns ---
        if self.shot_cooldown > 0: self.shot_cooldown -= 1
        if self.player_hit_cooldown > 0: self.player_hit_cooldown -= 1

        # --- Handle Actions ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_movement(movement)
        if space_held:
            self._handle_player_shooting()
        
        # --- Update Game World ---
        self._update_bullets()
        self._update_zombies()

        # --- Handle Collisions & Collect Rewards ---
        reward += self._handle_collisions()

        # --- Cleanup and Spawning ---
        self._cleanup_entities()
        self._spawn_periodic_entities()

        # --- Finalize Step ---
        self.steps += 1
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        
        if terminated:
            self.game_over = True
            if self.player_health <= 0:
                reward = -100.0  # Final penalty for dying
            elif self.steps >= self.MAX_STEPS:
                reward += 100.0 # Final bonus for survival
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_movement(self, movement_action):
        move_vector = np.array([0, 0], dtype=np.float32)
        if movement_action == 1: move_vector[1] -= 1 # Up
        elif movement_action == 2: move_vector[1] += 1 # Down
        elif movement_action == 3: move_vector[0] -= 1 # Left
        elif movement_action == 4: move_vector[0] += 1 # Right

        if np.linalg.norm(move_vector) > 0:
            norm_vector = move_vector / np.linalg.norm(move_vector)
            self.player_facing_direction = norm_vector
            self.player_pos += norm_vector * self.PLAYER_SPEED
        
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _handle_player_shooting(self):
        if self.shot_cooldown == 0 and self.player_ammo > 0:
            # SFX: Pew!
            self.player_ammo -= 1
            self.shot_cooldown = self.SHOT_COOLDOWN_STEPS
            bullet_pos = self.player_pos + self.player_facing_direction * (self.PLAYER_RADIUS + 1)
            bullet_vel = self.player_facing_direction * self.BULLET_SPEED
            self.bullets.append({'pos': bullet_pos, 'vel': bullet_vel})

    def _update_bullets(self):
        for b in self.bullets:
            b['pos'] += b['vel']

    def _update_zombies(self):
        for z in self.zombies:
            direction = self.player_pos - z['pos']
            dist = np.linalg.norm(direction)
            if dist > 1:
                z['pos'] += (direction / dist) * self.zombie_speed

    def _handle_collisions(self):
        reward = 0
        
        # Bullet-Zombie
        zombies_hit_indices = set()
        bullets_to_keep = []
        for b in self.bullets:
            hit_zombie = False
            for i, z in enumerate(self.zombies):
                if i in zombies_hit_indices: continue
                if np.linalg.norm(b['pos'] - z['pos']) < self.BULLET_RADIUS + self.ZOMBIE_RADIUS:
                    # SFX: Splat!
                    zombies_hit_indices.add(i)
                    self.score += 10
                    reward += 10.0
                    hit_zombie = True
                    break
            if not hit_zombie:
                bullets_to_keep.append(b)
        self.bullets = bullets_to_keep
        self.zombies = [z for i, z in enumerate(self.zombies) if i not in zombies_hit_indices]

        # Player-Zombie
        if self.player_hit_cooldown == 0:
            for z in self.zombies:
                if np.linalg.norm(self.player_pos - z['pos']) < self.PLAYER_RADIUS + self.ZOMBIE_RADIUS:
                    # SFX: Ouch!
                    self.player_health = max(0, self.player_health - 10)
                    self.player_hit_cooldown = self.PLAYER_HIT_COOLDOWN_STEPS
                    break
        
        # Player-Crate
        crates_to_keep = []
        for c in self.supply_crates:
            crate_rect = pygame.Rect(c['pos'][0] - self.CRATE_SIZE/2, c['pos'][1] - self.CRATE_SIZE/2, self.CRATE_SIZE, self.CRATE_SIZE)
            if crate_rect.collidepoint(self.player_pos):
                # SFX: Pickup!
                if c['type'] == 'health':
                    self.player_health = min(self.PLAYER_MAX_HEALTH, self.player_health + 25)
                    reward += 5.0
                elif c['type'] == 'ammo':
                    self.player_ammo = min(self.PLAYER_MAX_AMMO, self.player_ammo + 20)
                    reward += 5.0
                elif c['type'] == 'bonus':
                    self.score += 50
                    reward += 10.0
            else:
                crates_to_keep.append(c)
        self.supply_crates = crates_to_keep
        
        return reward

    def _cleanup_entities(self):
        self.bullets = [b for b in self.bullets if 0 < b['pos'][0] < self.WIDTH and 0 < b['pos'][1] < self.HEIGHT]

    def _spawn_periodic_entities(self):
        if self.steps > 0 and self.steps % self.CRATE_SPAWN_INTERVAL == 0:
            self._spawn_supply_crates()
        
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.zombie_speed += 0.1

    def _spawn_zombie(self):
        side = self.np_random.integers(4)
        if side == 0: # Top
            pos = [self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_RADIUS]
        elif side == 1: # Bottom
            pos = [self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_RADIUS]
        elif side == 2: # Left
            pos = [-self.ZOMBIE_RADIUS, self.np_random.uniform(0, self.HEIGHT)]
        else: # Right
            pos = [self.WIDTH + self.ZOMBIE_RADIUS, self.np_random.uniform(0, self.HEIGHT)]
        self.zombies.append({'pos': np.array(pos, dtype=np.float32)})

    def _spawn_supply_crates(self):
        self.supply_crates = []
        types = ['health', 'ammo', 'bonus']
        for t in types:
            pos = self.np_random.uniform([50, 50], [self.WIDTH - 50, self.HEIGHT - 50])
            self.supply_crates.append({'pos': pos.astype(np.float32), 'type': t})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_ARENA_BORDER, (0, 0, self.WIDTH, self.HEIGHT), 2)

        # Draw shadows
        for c in self.supply_crates: self._draw_shadow(c['pos'], self.CRATE_SIZE / 1.8)
        for z in self.zombies: self._draw_shadow(z['pos'], self.ZOMBIE_RADIUS)
        self._draw_shadow(self.player_pos, self.PLAYER_RADIUS)

        # Draw crates
        for c in self.supply_crates:
            color = self.COLOR_CRATE_HEALTH if c['type'] == 'health' else self.COLOR_CRATE_AMMO if c['type'] == 'ammo' else self.COLOR_CRATE_BONUS
            rect = pygame.Rect(c['pos'][0] - self.CRATE_SIZE/2, c['pos'][1] - self.CRATE_SIZE/2, self.CRATE_SIZE, self.CRATE_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=2)
            pygame.draw.rect(self.screen, tuple(x*0.7 for x in color), rect, 2, border_radius=2)

        # Draw zombies
        for z in self.zombies:
            pos_int = z['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ZOMBIE_RADIUS, self.COLOR_ZOMBIE)
        
        # Draw player
        player_color = self.COLOR_PLAYER if self.player_hit_cooldown % 4 < 2 else self.COLOR_PLAYER_HIT
        pos_int = self.player_pos.astype(int)
        for i in range(self.PLAYER_RADIUS, self.PLAYER_RADIUS + 5):
            alpha = 80 - (i - self.PLAYER_RADIUS) * 20
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], i, (*player_color, max(0, alpha)))
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, player_color)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, player_color)
        facing_end = self.player_pos + self.player_facing_direction * self.PLAYER_RADIUS * 0.7
        pygame.draw.aaline(self.screen, self.COLOR_BG, pos_int, facing_end.astype(int), 2)

        # Draw bullets
        for b in self.bullets:
            pos_int = b['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.BULLET_RADIUS, self.COLOR_BULLET)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BULLET_RADIUS, self.COLOR_BULLET)

    def _draw_shadow(self, pos, radius):
        shadow_pos = (int(pos[0] + 4), int(pos[1] + 4))
        pygame.gfxdraw.filled_ellipse(self.screen, shadow_pos[0], shadow_pos[1], int(radius), int(radius * 0.9), self.COLOR_SHADOW)

    def _render_ui(self):
        health_pct = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        bar_w, bar_h = 150, 20
        pygame.draw.rect(self.screen, self.COLOR_UI_DANGER, (10, 10, bar_w, bar_h), border_radius=3)
        if health_pct > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, int(bar_w * health_pct), bar_h), border_radius=3)
        
        ammo_text = self.font_ui.render(f"AMMO: {self.player_ammo}", True, self.COLOR_UI_TEXT)
        self.screen.blit(ammo_text, (10 + bar_w + 10, 12))
        
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_BONUS)
        self.screen.blit(score_text, score_text.get_rect(topright=(self.WIDTH - 10, 12)))
        
        time_left = max(0, self.MAX_STEPS - self.steps)
        minutes, seconds = divmod(time_left // 10, 60)
        timer_text = self.font_timer.render(f"{minutes:02}:{seconds:02}", True, self.COLOR_UI_DANGER)
        self.screen.blit(timer_text, timer_text.get_rect(midtop=(self.WIDTH / 2, 5)))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        # --- Human Input ---
        movement = 0 # None
        space = 0 # Released
        shift = 0 # Released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0

        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it.
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # The brief mentions 10 steps/sec. We'll run the game loop at that rate for human play.
        clock.tick(10)
        
    env.close()