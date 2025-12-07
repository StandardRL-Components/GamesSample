
# Generated: 2025-08-27T20:40:03.758450
# Source Brief: brief_02530.md
# Brief Index: 2530

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Hold Shift for a temporary shield. Press Space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down arcade shooter. Survive three stages of alien invaders and destroy them all to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS_PER_STAGE = 60 * self.FPS

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_SHIELD = (100, 150, 255, 100)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_ALIEN_S1 = (255, 80, 80)
        self.COLOR_ALIEN_S2 = (80, 120, 255)
        self.COLOR_ALIEN_S3 = (255, 255, 80)
        self.COLOR_UI = (220, 220, 220)
        self.ALIEN_COLORS = [self.COLOR_ALIEN_S1, self.COLOR_ALIEN_S2, self.COLOR_ALIEN_S3]

        # Player settings
        self.PLAYER_SIZE = 12
        self.PLAYER_SPEED = 5
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PLAYER_SHIELD_DURATION = 15 # frames
        self.PLAYER_SHIELD_COOLDOWN = 60 # frames
        self.PLAYER_PROJECTILE_SPEED = 10

        # Alien settings
        self.ALIEN_SIZE = 14
        self.ALIENS_PER_STAGE = [10, 15, 25] # Total 50
        self.ALIEN_BASE_SPEED = 1.0
        self.ALIEN_BASE_FIRE_RATE = 0.005 # chance per frame per alien

        # Gymnasium spaces
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_lives = None
        self.player_last_move_dir = None
        self.player_fire_cooldown_timer = None
        self.shield_active = None
        self.shield_timer = None
        self.shield_cooldown_timer = None
        self.aliens = None
        self.projectiles = None
        self.enemy_projectiles = None
        self.explosions = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.current_stage = None
        self.stage_timer = None
        self.rng = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            if self.rng is None:
                self.rng = np.random.default_rng()

        # Initialize all game state
        self.player_pos = pygame.math.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_vel = pygame.math.Vector2(0, 0)
        self.player_lives = 3
        self.player_last_move_dir = pygame.math.Vector2(0, -1) # Start aiming up
        self.player_fire_cooldown_timer = 0
        self.shield_active = False
        self.shield_timer = 0
        self.shield_cooldown_timer = 0

        self.aliens = []
        self.projectiles = []
        self.enemy_projectiles = []
        self.explosions = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.current_stage = 1
        self.stage_timer = self.MAX_STEPS_PER_STAGE
        self._spawn_aliens()

        return self._get_observation(), self._get_info()

    def step(self, action):
        step_reward = 0
        self.game_over = False

        # --- 1. Handle Input & Cooldowns ---
        self._handle_input(action)
        step_reward += self._update_shield(action)
        self._update_cooldowns()

        # --- 2. Update Game Logic ---
        self._update_player()
        step_reward += self._update_projectiles()
        self._update_aliens()
        step_reward += self._update_enemy_projectiles()
        self._update_explosions()

        # --- 3. Update Timers and Progression ---
        self.steps += 1
        self.stage_timer -= 1
        self._check_stage_completion()

        # --- 4. Check Termination Conditions ---
        terminated = self._check_termination()
        reward = step_reward
        if terminated:
            if self.game_won:
                reward += 100
            else:
                reward -= 100

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        move_map = {
            0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)
        }
        move_dir = pygame.math.Vector2(move_map[movement])
        
        if move_dir.length_squared() > 0:
            self.player_vel = move_dir.normalize() * self.PLAYER_SPEED
            self.player_last_move_dir = move_dir.normalize()
        else:
            self.player_vel = pygame.math.Vector2(0, 0)
        
        if space_held and self.player_fire_cooldown_timer == 0:
            # SFX: Player shoot
            proj_pos = self.player_pos + self.player_last_move_dir * (self.PLAYER_SIZE + 2)
            self.projectiles.append({
                "pos": proj_pos,
                "vel": self.player_last_move_dir * self.PLAYER_PROJECTILE_SPEED
            })
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_shield(self, action):
        _, _, shift_held = action
        reward = 0
        if shift_held and not self.shield_active and self.shield_cooldown_timer == 0:
            # SFX: Shield activate
            self.shield_active = True
            self.shield_timer = self.PLAYER_SHIELD_DURATION
            self.shield_cooldown_timer = self.PLAYER_SHIELD_COOLDOWN
            
            # Penalize safe shield usage
            safe_use = True
            for ep in self.enemy_projectiles:
                if self.player_pos.distance_to(ep["pos"]) < 150:
                    safe_use = False
                    break
            if safe_use:
                reward -= 0.02
        return reward

    def _update_cooldowns(self):
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        if self.shield_cooldown_timer > 0:
            self.shield_cooldown_timer -= 1
        if self.shield_timer > 0:
            self.shield_timer -= 1
        else:
            self.shield_active = False

    def _update_player(self):
        self.player_pos += self.player_vel
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        aliens_hit_indices = set()

        for proj in self.projectiles:
            proj["pos"] += proj["vel"]
            is_alive = True
            
            if not (0 < proj["pos"].x < self.WIDTH and 0 < proj["pos"].y < self.HEIGHT):
                is_alive = False

            if is_alive:
                proj_rect = pygame.Rect(proj["pos"].x - 2, proj["pos"].y - 2, 4, 4)
                for i, alien in enumerate(self.aliens):
                    if i in aliens_hit_indices:
                        continue
                    alien_rect = pygame.Rect(alien["pos"].x - self.ALIEN_SIZE/2, alien["pos"].y - self.ALIEN_SIZE/2, self.ALIEN_SIZE, self.ALIEN_SIZE)
                    if proj_rect.colliderect(alien_rect):
                        aliens_hit_indices.add(i)
                        is_alive = False
                        # SFX: Alien hit/explode
                        self._create_explosion(alien["pos"], 25, self.ALIEN_COLORS[self.current_stage-1])
                        self.score += 10
                        reward += 10.1 # +10 for kill, +0.1 for hit
                        break
            
            if is_alive:
                projectiles_to_keep.append(proj)
        
        self.projectiles = projectiles_to_keep
        if aliens_hit_indices:
            self.aliens = [alien for i, alien in enumerate(self.aliens) if i not in aliens_hit_indices]
        return reward

    def _update_aliens(self):
        fire_rate = self.ALIEN_BASE_FIRE_RATE + 0.01 * (self.current_stage - 1)
        speed = self.ALIEN_BASE_SPEED + 0.5 * (self.current_stage - 1)

        for alien in self.aliens:
            # Movement patterns
            if self.current_stage == 1: # Horizontal
                alien["pos"].x += alien["vel"].x * speed
                if alien["pos"].x < self.ALIEN_SIZE or alien["pos"].x > self.WIDTH - self.ALIEN_SIZE:
                    alien["vel"].x *= -1
            elif self.current_stage == 2: # Vertical
                alien["pos"].y += alien["vel"].y * speed
                if alien["pos"].y < self.ALIEN_SIZE or alien["pos"].y > self.HEIGHT / 2:
                    alien["vel"].y *= -1
            elif self.current_stage == 3: # Diagonal bounce
                alien["pos"] += alien["vel"] * speed
                if alien["pos"].x < self.ALIEN_SIZE or alien["pos"].x > self.WIDTH - self.ALIEN_SIZE:
                    alien["vel"].x *= -1
                if alien["pos"].y < self.ALIEN_SIZE or alien["pos"].y > self.HEIGHT / 2:
                    alien["vel"].y *= -1

            # Firing logic
            if self.rng.random() < fire_rate:
                # SFX: Alien shoot
                self.enemy_projectiles.append({
                    "pos": alien["pos"].copy(),
                    "vel": pygame.math.Vector2(0, 1) * (self.PLAYER_PROJECTILE_SPEED / 2)
                })

    def _update_enemy_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE/2, self.player_pos.y - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        for proj in self.enemy_projectiles:
            proj["pos"] += proj["vel"]
            is_alive = True
            
            if not (0 < proj["pos"].x < self.WIDTH and 0 < proj["pos"].y < self.HEIGHT):
                is_alive = False

            if is_alive:
                proj_rect = pygame.Rect(proj["pos"].x - 3, proj["pos"].y - 3, 6, 6)
                if proj_rect.colliderect(player_rect):
                    if self.shield_active:
                        # SFX: Shield block
                        self._create_explosion(proj["pos"], 15, self.COLOR_SHIELD)
                    else:
                        # SFX: Player hit
                        self.player_lives -= 1
                        reward -= 0.1
                        self._create_explosion(self.player_pos, 30, self.COLOR_PLAYER)
                    is_alive = False

            if is_alive:
                projectiles_to_keep.append(proj)

        self.enemy_projectiles = projectiles_to_keep
        return reward

    def _update_explosions(self):
        explosions_to_keep = []
        for exp in self.explosions:
            exp["radius"] += exp["speed"]
            if exp["radius"] < exp["max_radius"]:
                explosions_to_keep.append(exp)
        self.explosions = explosions_to_keep

    def _check_stage_completion(self):
        if not self.aliens:
            if self.current_stage < 3:
                self.current_stage += 1
                self.stage_timer = self.MAX_STEPS_PER_STAGE
                self._spawn_aliens()
                # SFX: Stage complete
            else:
                self.game_won = True
                self.game_over = True

    def _check_termination(self):
        if self.player_lives <= 0 or self.stage_timer <= 0:
            self.game_over = True
        return self.game_over

    def _spawn_aliens(self):
        num_aliens = self.ALIENS_PER_STAGE[self.current_stage - 1]
        rows = 2 if self.current_stage == 1 else 3
        cols = (num_aliens + rows -1) // rows
        
        for i in range(num_aliens):
            row = i // cols
            col = i % cols
            x = (self.WIDTH / (cols + 1)) * (col + 1)
            y = 50 + row * 40
            
            if self.current_stage == 1: vel = pygame.math.Vector2(1, 0)
            elif self.current_stage == 2: vel = pygame.math.Vector2(0, 1)
            else: vel = pygame.math.Vector2(self.rng.choice([-1, 1]), self.rng.choice([-1, 1])).normalize()

            self.aliens.append({"pos": pygame.math.Vector2(x, y), "vel": vel})

    def _create_explosion(self, pos, max_radius, color):
        self.explosions.append({
            "pos": pos.copy(),
            "radius": 0,
            "max_radius": max_radius,
            "speed": max(1, max_radius / 15),
            "color": color
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Explosions
        for exp in self.explosions:
            alpha = 255 * (1 - exp["radius"] / exp["max_radius"])
            pygame.gfxdraw.filled_circle(
                self.screen, int(exp["pos"].x), int(exp["pos"].y), int(exp["radius"]),
                (*exp["color"], int(alpha))
            )

        # Player projectiles
        for proj in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(proj["pos"].x), int(proj["pos"].y)), 2)

        # Enemy projectiles
        for proj in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(proj["pos"].x), int(proj["pos"].y), 3, self.ALIEN_COLORS[self.current_stage-1])

        # Aliens
        for alien in self.aliens:
            color = self.ALIEN_COLORS[self.current_stage - 1]
            rect = pygame.Rect(alien["pos"].x - self.ALIEN_SIZE/2, alien["pos"].y - self.ALIEN_SIZE/2, self.ALIEN_SIZE, self.ALIEN_SIZE)
            pygame.draw.rect(self.screen, color, rect, border_radius=2)

        # Player
        if self.player_lives > 0:
            angle = self.player_last_move_dir.angle_to(pygame.math.Vector2(0, -1))
            points = [
                pygame.math.Vector2(0, -self.PLAYER_SIZE).rotate(-angle) + self.player_pos,
                pygame.math.Vector2(-self.PLAYER_SIZE/1.5, self.PLAYER_SIZE/2).rotate(-angle) + self.player_pos,
                pygame.math.Vector2(self.PLAYER_SIZE/1.5, self.PLAYER_SIZE/2).rotate(-angle) + self.player_pos,
            ]
            int_points = [(int(p.x), int(p.y)) for p in points]
            
            # Glow effect
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER_GLOW)
            
            # Main ship
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

        # Shield
        if self.shield_active:
            radius = self.PLAYER_SIZE + 5
            alpha = self.COLOR_SHIELD[3] * (self.shield_timer / self.PLAYER_SHIELD_DURATION)
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), radius, (*self.COLOR_SHIELD[:3], int(alpha)))
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), radius, (*self.COLOR_SHIELD[:3], int(alpha*1.5)))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_small.render(f"LIVES: {self.player_lives}", True, self.COLOR_UI)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Stage
        stage_text = self.font_small.render(f"STAGE: {self.current_stage}", True, self.COLOR_UI)
        self.screen.blit(stage_text, (self.WIDTH/2 - stage_text.get_width()/2, 10))

        # Stage Timer Bar
        timer_ratio = self.stage_timer / self.MAX_STEPS_PER_STAGE
        pygame.draw.rect(self.screen, self.COLOR_UI, (self.WIDTH/2 - 100, 35, 200, 5))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (self.WIDTH/2 - 100, 35, 200 * timer_ratio, 5))

        # Game Over / Win Text
        if self.game_over:
            text_str = "GAME OVER" if not self.game_won else "YOU WIN!"
            color = self.COLOR_ALIEN_S1 if not self.game_won else self.COLOR_PLAYER
            end_text = self.font_large.render(text_str, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "stage": self.current_stage,
            "game_won": self.game_won
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Shooter")
    clock = pygame.time.Clock()

    done = False
    total_reward = 0
    
    # --- Human Controls ---
    # Map keyboard keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    while not done:
        # Action defaults
        movement_action = 0 # no-op
        space_action = 0    # released
        shift_action = 0    # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            done = True
        
        # Check movement keys
        for key, move_val in key_map.items():
            if keys[key]:
                movement_action = move_val
                break # Prioritize first key found (e.g., up over down)

        # Check action keys
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
        
        action = [movement_action, space_action, shift_action]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()