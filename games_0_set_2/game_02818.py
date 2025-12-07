
# Generated: 2025-08-27T21:31:52.874519
# Source Brief: brief_02818.md
# Brief Index: 2818

        
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
        "Controls: Arrow keys to move. Hold Space to fire your weapon. Survive the onslaught!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, top-down arcade shooter. Survive against ever-growing waves of the undead horde."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Game Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 45, 50)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_FACING = (255, 255, 255)
    COLOR_ZOMBIE = (255, 50, 50)
    COLOR_ZOMBIE_DAMAGED = (180, 40, 40)
    COLOR_PROJECTILE = (255, 255, 100)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_HEALTH_FG = (255, 60, 60)
    COLOR_UI_HEALTH_BG = (70, 70, 70)
    
    # Player
    PLAYER_SIZE = 12
    PLAYER_SPEED = 4.0
    PLAYER_HEALTH_MAX = 100
    PLAYER_DAMAGE_COOLDOWN = 30 # frames of invincibility
    
    # Weapon
    PROJECTILE_SIZE = (4, 12)
    PROJECTILE_SPEED = 10.0
    SHOOT_COOLDOWN_MAX = 8 # frames between shots
    
    # Zombies
    ZOMBIE_SIZE = 10
    ZOMBIE_HEALTH_MAX = 20
    ZOMBIE_DAMAGE = 10
    ZOMBIE_BASE_COUNT = 20
    ZOMBIE_COUNT_INCREMENT = 4
    ZOMBIE_BASE_SPEED = 0.5
    ZOMBIE_SPEED_INCREMENT = 0.05
    
    # Game
    MAX_WAVES = 5
    MAX_EPISODE_STEPS = 10000
    GRID_SPACING = 40

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 50)
        self.font_medium = pygame.font.Font(None, 32)
        
        self.np_random = None
        
        # Initialize state variables that are not reset
        self.space_was_held = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_facing_direction = pygame.Vector2(0, -1) # Default up
        self.player_damage_timer = 0

        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.current_wave = 0
        self.shoot_cooldown = 0
        self.space_was_held = False
        
        self._start_new_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01 # Cost of living
        
        if not self.game_over:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            self._handle_player_input(movement, space_held)
            reward += self._update_game_state()
            self._update_timers()

        self.steps += 1
        
        if self.player_health <= 0:
            self.game_over = True
        
        if self.steps >= self.MAX_EPISODE_STEPS:
            self.game_over = True
            
        terminated = self.game_over
        
        # Final win reward
        if self.game_won:
             reward += 500
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
        
    def _start_new_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            self.game_won = True
            self.game_over = True
            return

        num_zombies = self.ZOMBIE_BASE_COUNT + (self.current_wave - 1) * self.ZOMBIE_COUNT_INCREMENT
        zombie_speed = self.ZOMBIE_BASE_SPEED + (self.current_wave - 1) * self.ZOMBIE_SPEED_INCREMENT

        for _ in range(num_zombies):
            self._spawn_zombie(zombie_speed)
            
    def _spawn_zombie(self, speed):
        edge = self.np_random.integers(4)
        if edge == 0: # Top
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), -self.ZOMBIE_SIZE)
        elif edge == 1: # Bottom
            pos = pygame.Vector2(self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + self.ZOMBIE_SIZE)
        elif edge == 2: # Left
            pos = pygame.Vector2(-self.ZOMBIE_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT))
        else: # Right
            pos = pygame.Vector2(self.SCREEN_WIDTH + self.ZOMBIE_SIZE, self.np_random.uniform(0, self.SCREEN_HEIGHT))
            
        self.zombies.append({
            "pos": pos,
            "speed": speed,
            "health": self.ZOMBIE_HEALTH_MAX,
            "max_health": self.ZOMBIE_HEALTH_MAX
        })

    def _handle_player_input(self, movement, space_held):
        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right

        if move_vec.length() > 0:
            self.player_facing_direction = move_vec.normalize()
            self.player_pos += self.player_facing_direction * self.PLAYER_SPEED
        
        # Keep player in bounds
        self.player_pos.x = max(self.PLAYER_SIZE, min(self.player_pos.x, self.SCREEN_WIDTH - self.PLAYER_SIZE))
        self.player_pos.y = max(self.PLAYER_SIZE, min(self.player_pos.y, self.SCREEN_HEIGHT - self.PLAYER_SIZE))

        # Shooting
        if space_held and self.shoot_cooldown <= 0:
            # sfx: player_shoot.wav
            self.projectiles.append({
                "pos": self.player_pos.copy(),
                "dir": self.player_facing_direction.copy()
            })
            self.shoot_cooldown = self.SHOOT_COOLDOWN_MAX

    def _update_game_state(self):
        step_reward = 0
        self._update_projectiles()
        self._update_zombies()
        step_reward += self._handle_collisions()
        self._update_particles()
        
        # Wave clear check
        if not self.zombies and not self.game_won:
            # sfx: wave_clear.wav
            step_reward += 100
            self._start_new_wave()
            
        return step_reward

    def _update_timers(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.player_damage_timer > 0:
            self.player_damage_timer -= 1
            
    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"] += p["dir"] * self.PROJECTILE_SPEED
            if not self.screen.get_rect().collidepoint(p["pos"]):
                self.projectiles.remove(p)
                
    def _update_zombies(self):
        for z in self.zombies:
            direction = (self.player_pos - z["pos"])
            if direction.length() > 0:
                z["pos"] += direction.normalize() * z["speed"]
                
    def _handle_collisions(self):
        collision_reward = 0
        
        # Projectiles vs Zombies
        for p in self.projectiles[:]:
            proj_rect = pygame.Rect(p["pos"].x - self.PROJECTILE_SIZE[0]/2, p["pos"].y - self.PROJECTILE_SIZE[1]/2, *self.PROJECTILE_SIZE)
            if p["dir"].x != 0: proj_rect.size = (self.PROJECTILE_SIZE[1], self.PROJECTILE_SIZE[0])
            
            for z in self.zombies[:]:
                if z["pos"].distance_to(p["pos"]) < self.ZOMBIE_SIZE + max(self.PROJECTILE_SIZE):
                    # sfx: zombie_hit.wav
                    z["health"] -= 10
                    collision_reward += 0.1
                    self._create_particles(p["pos"], 5, self.COLOR_PROJECTILE)
                    if p in self.projectiles: self.projectiles.remove(p)
                    
                    if z["health"] <= 0:
                        # sfx: zombie_die.wav
                        self.score += 10
                        collision_reward += 1.0
                        self._create_particles(z["pos"], 20, self.COLOR_ZOMBIE)
                        if z in self.zombies: self.zombies.remove(z)
                    break

        # Player vs Zombies
        if self.player_damage_timer <= 0:
            for z in self.zombies:
                if z["pos"].distance_to(self.player_pos) < self.ZOMBIE_SIZE + self.PLAYER_SIZE:
                    # sfx: player_hurt.wav
                    self.player_health -= self.ZOMBIE_DAMAGE
                    self.player_damage_timer = self.PLAYER_DAMAGE_COOLDOWN
                    self._create_particles(self.player_pos, 15, self.COLOR_PLAYER)
                    # Small knockback
                    knockback = (self.player_pos - z["pos"]).normalize() * 15
                    self.player_pos += knockback
                    break

        return collision_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            self.particles.append({
                "pos": pos.copy(),
                "vel": pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
                "life": self.np_random.integers(10, 20),
                "color": color,
                "size": self.np_random.uniform(1, 4)
            })
            
    def _get_observation(self):
        self._draw_background()
        self._draw_entities()
        self._draw_effects()
        self._draw_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_background(self):
        self.screen.fill(self.COLOR_BG)
        for x in range(0, self.SCREEN_WIDTH, self.GRID_SPACING):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.GRID_SPACING):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_entities(self):
        # Draw zombies
        for z in self.zombies:
            color = self.COLOR_ZOMBIE if z["health"] == z["max_health"] else self.COLOR_ZOMBIE_DAMAGED
            pygame.gfxdraw.filled_circle(self.screen, int(z["pos"].x), int(z["pos"].y), self.ZOMBIE_SIZE, color)
            pygame.gfxdraw.aacircle(self.screen, int(z["pos"].x), int(z["pos"].y), self.ZOMBIE_SIZE, color)

        # Draw projectiles
        for p in self.projectiles:
            angle = p["dir"].angle_to(pygame.Vector2(0, -1))
            rotated_surf = pygame.transform.rotate(
                pygame.Surface(self.PROJECTILE_SIZE, pygame.SRCALPHA), angle
            )
            rotated_surf.fill(self.COLOR_PROJECTILE)
            rect = rotated_surf.get_rect(center=p["pos"])
            self.screen.blit(rotated_surf, rect)

        # Draw player (with invincibility flicker)
        if self.player_damage_timer % 6 < 3:
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE, self.COLOR_PLAYER)
            
            # Draw facing indicator
            p2 = self.player_pos + self.player_facing_direction * self.PLAYER_SIZE
            pygame.draw.line(self.screen, self.COLOR_PLAYER_FACING, self.player_pos, p2, 3)

    def _draw_effects(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 20))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, p["pos"] - pygame.Vector2(p["size"], p["size"]))

    def _draw_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / self.PLAYER_HEALTH_MAX)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_FG, (10, 10, int(bar_width * health_pct), bar_height))
        
        # Wave Text
        wave_text = self.font_medium.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))
        
        # Score Text
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 40))

        # Game Over / Win Text
        if self.game_over:
            if self.game_won:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "player_health": self.player_health,
            "zombies_remaining": len(self.zombies)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Zombie Survival")
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS
        
    env.close()