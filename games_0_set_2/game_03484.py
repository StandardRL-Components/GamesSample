import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to fire your weapon. Survive the onslaught!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A one-eyed gunner must survive a 60-second bullet hell. Dodge enemy fire and blast them to bits to score points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.MAX_STEPS = self.FPS * 60  # 60 seconds

        # Colors
        self.COLOR_BG = (10, 5, 15)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_PLAYER_GLOW = (200, 200, 255)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (200, 50, 50)
        self.COLOR_PLAYER_BULLET = (150, 255, 255)
        self.COLOR_ENEMY_BULLET = (255, 150, 50)
        self.COLOR_MUZZLE_FLASH = (255, 255, 100)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_HEART = (255, 0, 100)

        # Player
        self.PLAYER_SPEED = 4.0
        self.PLAYER_RADIUS = 10
        self.PLAYER_MAX_HEALTH = 3
        self.PLAYER_SHOOT_COOLDOWN = 6 # frames

        # Enemy
        self.ENEMY_RADIUS = 12
        self.ENEMY_SPEED = 1.0
        self.ENEMY_SHOOT_COOLDOWN = 45 # frames

        # Bullets
        self.PLAYER_BULLET_SPEED = 8.0
        self.PLAYER_BULLET_RADIUS = 3
        self.ENEMY_BULLET_SPEED_BASE = 2.5
        self.ENEMY_BULLET_RADIUS = 4
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_score = pygame.font.Font(None, 48)
        
        # --- Game State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_health = 0
        self.player_shoot_cooldown_timer = 0
        self.player_aim_angle = 0.0
        self.last_movement_vector = pygame.Vector2(1, 0)
        self.player_invincibility_timer = 0
        
        self.enemies = []
        self.player_bullets = []
        self.enemy_bullets = []
        self.particles = []
        
        self.spawn_rate = 0.0
        self.enemy_bullet_speed = 0.0
        self.difficulty_timer = 0

        # Run validation
        # self.validate_implementation() # Commented out for submission, but useful for local dev
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_shoot_cooldown_timer = 0
        self.player_aim_angle = 0.0
        self.last_movement_vector = pygame.Vector2(1, 0)
        self.player_invincibility_timer = self.FPS * 2 # 2 seconds of spawn invincibility

        self.enemies.clear()
        self.player_bullets.clear()
        self.enemy_bullets.clear()
        self.particles.clear()
        
        self.spawn_rate = 0.02 # spawns per frame
        self.enemy_bullet_speed = self.ENEMY_BULLET_SPEED_BASE
        self.difficulty_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.001 # Small penalty for existing, encourages faster completion
        
        # --- Action Handling ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._update_player(movement, space_held)

        # --- Game Logic Updates ---
        self._update_difficulty()
        self._update_enemies()
        self._update_bullets()
        self._update_particles()
        
        # --- Collision Detection & Rewards ---
        hit_reward, destroy_reward = self._handle_collisions()
        reward += hit_reward + destroy_reward
        
        # --- Termination Check ---
        self.steps += 1
        terminated = False
        if self.player_health <= 0:
            reward -= 100 # Large penalty for losing
            terminated = True
            # sfx: player_death
        elif self.steps >= self.MAX_STEPS:
            reward += 100 # Large reward for winning
            terminated = True
            # sfx: game_win
        
        if terminated:
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement, space_held):
        # Update timers
        if self.player_shoot_cooldown_timer > 0:
            self.player_shoot_cooldown_timer -= 1
        if self.player_invincibility_timer > 0:
            self.player_invincibility_timer -= 1
            
        # --- Movement ---
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length_squared() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.last_movement_vector = pygame.Vector2(move_vec)
        
        self.player_aim_angle = self.last_movement_vector.angle_to(pygame.Vector2(1, 0))

        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

        # --- Shooting ---
        if space_held and self.player_shoot_cooldown_timer == 0:
            self.player_shoot_cooldown_timer = self.PLAYER_SHOOT_COOLDOWN
            
            # Create muzzle flash
            flash_pos = self.player_pos + self.last_movement_vector * (self.PLAYER_RADIUS + 5)
            self._create_particles(1, flash_pos, self.COLOR_MUZZLE_FLASH, 3, 15, 0)
            
            # Create bullet
            bullet_vel = self.last_movement_vector * self.PLAYER_BULLET_SPEED
            self.player_bullets.append({
                "pos": pygame.Vector2(flash_pos),
                "vel": bullet_vel
            })
            # sfx: player_shoot

    def _update_difficulty(self):
        self.difficulty_timer += 1
        if self.difficulty_timer >= self.FPS * 10: # Every 10 seconds
            self.difficulty_timer = 0
            self.spawn_rate += 0.005
            self.enemy_bullet_speed = min(self.enemy_bullet_speed + 0.2, self.PLAYER_SPEED - 0.5)

    def _update_enemies(self):
        # Spawn new enemies
        if self.np_random.random() < self.spawn_rate:
            self._spawn_enemy()
        
        # Update existing enemies
        for enemy in self.enemies:
            # Movement
            vec_to_player = self.player_pos - enemy["pos"]
            if vec_to_player.length_squared() > 0:
                enemy["vel"] = vec_to_player.normalize() * self.ENEMY_SPEED
            enemy["pos"] += enemy["vel"]
            
            # Shooting
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0:
                enemy["shoot_cooldown"] = self.ENEMY_SHOOT_COOLDOWN + self.np_random.integers(-10, 10)
                bullet_vel = (self.player_pos - enemy["pos"]).normalize() * self.enemy_bullet_speed
                self.enemy_bullets.append({
                    "pos": pygame.Vector2(enemy["pos"]),
                    "vel": bullet_vel
                })
                # sfx: enemy_shoot

    def _spawn_enemy(self):
        edge = self.np_random.integers(0, 4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -self.ENEMY_RADIUS)
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ENEMY_RADIUS)
        elif edge == 2: # Left
            pos = pygame.Vector2(-self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT))
        else: # Right
            pos = pygame.Vector2(self.WIDTH + self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT))
        
        self.enemies.append({
            "pos": pos,
            "vel": pygame.Vector2(0, 0),
            "shoot_cooldown": self.np_random.integers(30, self.FPS * 2)
        })

    def _update_bullets(self):
        # Player bullets
        for bullet in self.player_bullets:
            bullet["pos"] += bullet["vel"]
        self.player_bullets = [b for b in self.player_bullets if self.screen.get_rect().collidepoint(b["pos"])]
        
        # Enemy bullets
        for bullet in self.enemy_bullets:
            bullet["pos"] += bullet["vel"]
        self.enemy_bullets = [b for b in self.enemy_bullets if self.screen.get_rect().collidepoint(b["pos"])]

    def _handle_collisions(self):
        hit_reward = 0
        destroy_reward = 0
        
        # Player bullets vs Enemies
        for p_bullet in self.player_bullets[:]:
            for enemy in self.enemies[:]:
                if p_bullet["pos"].distance_to(enemy["pos"]) < self.PLAYER_BULLET_RADIUS + self.ENEMY_RADIUS:
                    self.enemies.remove(enemy)
                    if p_bullet in self.player_bullets: self.player_bullets.remove(p_bullet)
                    self.score += 10
                    destroy_reward += 1
                    self._create_particles(20, enemy["pos"], self.COLOR_ENEMY, 20, 5, 2)
                    # sfx: enemy_explode
                    break
        
        # Enemy bullets vs Player
        if self.player_invincibility_timer == 0:
            for e_bullet in self.enemy_bullets[:]:
                if e_bullet["pos"].distance_to(self.player_pos) < self.ENEMY_BULLET_RADIUS + self.PLAYER_RADIUS:
                    self.enemy_bullets.remove(e_bullet)
                    self.player_health -= 1
                    hit_reward -= 10
                    self.player_invincibility_timer = self.FPS # 1 second invincibility
                    self._create_particles(30, self.player_pos, self.COLOR_PLAYER, 15, 4, 1)
                    # sfx: player_hit
                    break
        return hit_reward, destroy_reward

    def _update_particles(self):
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0]

    def _create_particles(self, count, pos, color, max_life, max_speed, spread):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            vel += pygame.Vector2(self.np_random.uniform(-spread, spread), self.np_random.uniform(-spread, spread))
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "life": self.np_random.integers(max_life // 2, max_life),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
        }

    def _render_game(self):
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            if alpha > 0:
                size = int(p["life"] / 4)
                if size > 0:
                    pygame.draw.circle(self.screen, p["color"], (int(p["pos"].x), int(p["pos"].y)), size, 0)

        # Enemy Bullets
        for bullet in self.enemy_bullets:
            pos = (int(bullet["pos"].x), int(bullet["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_BULLET_RADIUS, self.COLOR_ENEMY_BULLET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_BULLET_RADIUS, self.COLOR_ENEMY_BULLET)

        # Player Bullets
        for bullet in self.player_bullets:
            pos = (int(bullet["pos"].x), int(bullet["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_BULLET_RADIUS, self.COLOR_PLAYER_BULLET)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_BULLET_RADIUS, self.COLOR_PLAYER_BULLET)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, self.COLOR_ENEMY_GLOW)
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS-3, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS-3, self.COLOR_ENEMY)

        # Player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Invincibility flash
        if self.player_invincibility_timer > 0 and (self.steps // 3) % 2 == 0:
            pass # Don't render player to make it flash
        else:
            # Glow
            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER_GLOW)
            # Body
            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS-2, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS-2, self.COLOR_PLAYER)
            # Eye/Gun
            aim_vec = pygame.Vector2(1, 0).rotate(-self.player_aim_angle)
            eye_pos = self.player_pos + aim_vec * (self.PLAYER_RADIUS * 0.5)
            pygame.draw.line(self.screen, self.COLOR_BG, player_pos_int, (int(eye_pos.x), int(eye_pos.y)), 3)

    def _render_ui(self):
        # Score
        score_text = self.font_score.render(f"{self.score:05d}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 30))
        self.screen.blit(score_text, score_rect)
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH - 120, 10))
        
        # Health
        for i in range(self.PLAYER_MAX_HEALTH):
            heart_pos = (20 + i * 30, 25)
            if i < self.player_health:
                color = self.COLOR_HEART
            else:
                color = (50, 50, 50)
            pygame.gfxdraw.filled_polygon(self.screen, [
                (heart_pos[0], heart_pos[1]-5),
                (heart_pos[0]-10, heart_pos[1]-15),
                (heart_pos[0]-10, heart_pos[1]-5),
                (heart_pos[0], heart_pos[1]+5),
                (heart_pos[0]+10, heart_pos[1]-5),
                (heart_pos[0]+10, heart_pos[1]-15),
            ], color)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space (after a reset)
        self.reset()
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
    # This block allows you to play the game directly
    # pip install gymnasium[pygame]
    
    # Re-enable display for direct play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Bullet Hell Survivor")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)

    while running:
        movement = 0 # No-op
        space_held = 0
        
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
        
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for 'r' to reset or 'q' to quit
            wait_for_input = True
            while wait_for_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        running = False
                        wait_for_input = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print("--- Game Reset ---")
                        wait_for_input = False
        
        clock.tick(env.FPS)
        
    env.close()