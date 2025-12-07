
# Generated: 2025-08-28T04:46:09.404870
# Source Brief: brief_05356.md
# Brief Index: 5356

        
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

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot. Press Shift to reload."
    )

    # User-facing description of the game
    game_description = (
        "Survive waves of procedurally generated zombies in a top-down arena shooter. "
        "Manage your ammo and position to avoid being overwhelmed."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 5000
        self.MAX_WAVES = 20
        self.WALL_THICKNESS = 10

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_WALL = (60, 65, 70)
        self.COLOR_PLAYER = (50, 200, 50)
        self.COLOR_ZOMBIE = (200, 50, 50)
        self.COLOR_BULLET = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_MUZZLE_FLASH = (255, 220, 150)

        # Player properties
        self.PLAYER_SIZE = 10
        self.PLAYER_SPEED = 3.0
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_MAX_AMMO = 30
        self.SHOOT_COOLDOWN = 5  # frames
        self.RELOAD_TIME = 60  # frames

        # Zombie properties
        self.ZOMBIE_SIZE = 12
        self.ZOMBIE_HEALTH = 25
        self.ZOMBIE_DAMAGE = 10
        self.ZOMBIE_BASE_SPEED = 0.8

        # Bullet properties
        self.BULLET_SIZE = 4
        self.BULLET_SPEED = 10.0
        self.BULLET_DAMAGE = 25

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_wave = pygame.font.SysFont("monospace", 24, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_ammo = None
        self.player_facing_angle = None
        self.shoot_cooldown = None
        self.reload_cooldown = None
        self.zombies = None
        self.bullets = None
        self.particles = None
        self.wave = None
        self.steps = None
        self.score = None
        self.muzzle_flash_timer = None
        
        self.reset()
        
        # self.validate_implementation() # Optional: call to check against spec

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_ammo = self.PLAYER_MAX_AMMO
        self.player_facing_angle = -math.pi / 2  # Start facing up
        self.shoot_cooldown = 0
        self.reload_cooldown = 0

        self.zombies = []
        self.bullets = []
        self.particles = []
        
        self.wave = 1
        self._spawn_wave()

        self.steps = 0
        self.score = 0
        self.muzzle_flash_timer = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        # Decrement cooldowns and timers
        self._update_timers()

        # Handle player input
        reward += self._handle_input(action)
        
        # Update game objects
        self._update_player()
        self._update_bullets()
        reward += self._update_zombies()
        self._update_particles()
        
        # Check for wave completion
        if not self.zombies:
            reward += 10.0  # Wave clear bonus
            if self.wave >= self.MAX_WAVES:
                terminated = True # Game won
                reward += 100.0 # Victory bonus
            else:
                self.wave += 1
                self._spawn_wave()
                # sound: wave_start.wav

        # Check termination conditions
        if self.player_health <= 0:
            terminated = True
            reward = -100.0  # Death penalty
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.steps += 1
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated
            self._get_info(),
        )

    def _update_timers(self):
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
        if self.reload_cooldown > 0:
            self.reload_cooldown -= 1
        if self.muzzle_flash_timer > 0:
            self.muzzle_flash_timer -= 1
            
    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1:  # Up
            move_vec[1] -= 1
        elif movement == 2:  # Down
            move_vec[1] += 1
        elif movement == 3:  # Left
            move_vec[0] -= 1
        elif movement == 4:  # Right
            move_vec[0] += 1

        if np.linalg.norm(move_vec) > 0:
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.player_facing_angle = math.atan2(move_vec[1], move_vec[0])

        if space_held and self.shoot_cooldown == 0 and self.reload_cooldown == 0 and self.player_ammo > 0:
            self._shoot()
            # sound: shoot.wav

        if shift_held and self.reload_cooldown == 0 and self.player_ammo < self.PLAYER_MAX_AMMO:
            self._reload()
            # sound: reload.wav
            
        return reward

    def _shoot(self):
        self.player_ammo -= 1
        self.shoot_cooldown = self.SHOOT_COOLDOWN
        self.muzzle_flash_timer = 2
        
        vel = np.array([math.cos(self.player_facing_angle), math.sin(self.player_facing_angle)]) * self.BULLET_SPEED
        # Spawn bullet slightly in front of the player
        start_pos = self.player_pos + np.array([math.cos(self.player_facing_angle), math.sin(self.player_facing_angle)]) * (self.PLAYER_SIZE + 1)
        
        self.bullets.append({
            "pos": start_pos,
            "vel": vel,
        })

    def _reload(self):
        self.reload_cooldown = self.RELOAD_TIME
        self.player_ammo = self.PLAYER_MAX_AMMO
        
    def _update_player(self):
        # Clamp player position to stay within arena bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.WALL_THICKNESS + self.PLAYER_SIZE, self.WIDTH - self.WALL_THICKNESS - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.WALL_THICKNESS + self.PLAYER_SIZE, self.HEIGHT - self.WALL_THICKNESS - self.PLAYER_SIZE)

    def _update_bullets(self):
        miss_penalty = 0
        for bullet in self.bullets[:]:
            bullet["pos"] += bullet["vel"]
            # Remove bullets that go off-screen
            if not (0 < bullet["pos"][0] < self.WIDTH and 0 < bullet["pos"][1] < self.HEIGHT):
                self.bullets.remove(bullet)
                miss_penalty -= 0.01 # Miss penalty
        return miss_penalty
        
    def _update_zombies(self):
        kill_reward = 0
        hit_reward = 0
        
        # Zombie-player collision
        for zombie in self.zombies:
            dist = np.linalg.norm(self.player_pos - zombie["pos"])
            if dist < self.PLAYER_SIZE + self.ZOMBIE_SIZE:
                self.player_health -= self.ZOMBIE_DAMAGE
                self.player_health = max(0, self.player_health)
                # Simple knockback
                knockback_vec = (self.player_pos - zombie["pos"]) / (dist + 1e-6)
                self.player_pos += knockback_vec * 5
                # sound: player_hit.wav
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 5, 2.0)


        # Bullet-zombie collision and zombie movement
        current_zombie_speed = self.ZOMBIE_BASE_SPEED + (self.wave - 1) * 0.05
        for zombie in self.zombies[:]:
            # Movement
            direction = self.player_pos - zombie["pos"]
            dist_to_player = np.linalg.norm(direction)
            if dist_to_player > 1: # Avoid division by zero
                direction /= dist_to_player
                zombie["pos"] += direction * current_zombie_speed
            
            # Collision with bullets
            for bullet in self.bullets[:]:
                if np.linalg.norm(zombie["pos"] - bullet["pos"]) < self.ZOMBIE_SIZE + self.BULLET_SIZE:
                    zombie["health"] -= self.BULLET_DAMAGE
                    self._create_particles(zombie["pos"], self.COLOR_ZOMBIE, 10, 3.0)
                    self.bullets.remove(bullet)
                    hit_reward += 0.1
                    # sound: zombie_hit.wav
                    
                    if zombie["health"] <= 0:
                        if zombie in self.zombies: # Check if not already removed
                            self.zombies.remove(zombie)
                        kill_reward += 1.0
                        self._create_particles(zombie["pos"], self.COLOR_ZOMBIE, 30, 4.0)
                        # sound: zombie_death.wav
                    break # A bullet can only hit one zombie
                    
        return kill_reward + hit_reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        num_zombies = 3 + self.wave * 2
        for _ in range(num_zombies):
            while True:
                pos = np.array([
                    self.np_random.uniform(self.WALL_THICKNESS, self.WIDTH - self.WALL_THICKNESS),
                    self.np_random.uniform(self.WALL_THICKNESS, self.HEIGHT - self.WALL_THICKNESS)
                ])
                if np.linalg.norm(pos - self.player_pos) > 150: # Spawn away from player
                    break
            self.zombies.append({"pos": pos, "health": self.ZOMBIE_HEALTH})
            
    def _create_particles(self, pos, color, count, speed_scale):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5) * speed_scale
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": self.np_random.integers(10, 20),
                "color": color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw walls
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, self.HEIGHT - self.WALL_THICKNESS, self.WIDTH, self.WALL_THICKNESS))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.WALL_THICKNESS, self.HEIGHT))
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.WIDTH - self.WALL_THICKNESS, 0, self.WALL_THICKNESS, self.HEIGHT))

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], p["pos"].astype(int), int(p["life"] / 5))

        # Draw zombies
        for zombie in self.zombies:
            pos_int = zombie["pos"].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ZOMBIE_SIZE, self.COLOR_ZOMBIE)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ZOMBIE_SIZE, self.COLOR_ZOMBIE)

        # Draw player
        player_pos_int = self.player_pos.astype(int)
        pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_SIZE, self.COLOR_PLAYER)
        
        # Draw facing indicator
        facing_end_pos = self.player_pos + np.array([math.cos(self.player_facing_angle), math.sin(self.player_facing_angle)]) * (self.PLAYER_SIZE + 2)
        pygame.draw.line(self.screen, self.COLOR_PLAYER, player_pos_int, facing_end_pos.astype(int), 3)

        # Draw bullets
        for bullet in self.bullets:
            pos_int = bullet["pos"].astype(int)
            pygame.draw.circle(self.screen, self.COLOR_BULLET, pos_int, self.BULLET_SIZE)

        # Draw muzzle flash
        if self.muzzle_flash_timer > 0:
            flash_pos = self.player_pos + np.array([math.cos(self.player_facing_angle), math.sin(self.player_facing_angle)]) * (self.PLAYER_SIZE + 8)
            pygame.draw.circle(self.screen, self.COLOR_MUZZLE_FLASH, flash_pos.astype(int), 8)

    def _render_ui(self):
        # Health display
        health_text = self.font_ui.render(f"HP: {self.player_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, self.HEIGHT - 30))

        # Ammo display
        if self.reload_cooldown > 0:
            ammo_text_str = "RELOADING"
        else:
            ammo_text_str = f"AMMO: {self.player_ammo}/{self.PLAYER_MAX_AMMO}"
        ammo_text = self.font_ui.render(ammo_text_str, True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (120, self.HEIGHT - 30))
        
        # Wave display
        wave_text = self.font_wave.render(f"WAVE {self.wave}", True, self.COLOR_TEXT)
        text_rect = wave_text.get_rect(center=(self.WIDTH / 2, 30))
        self.screen.blit(wave_text, text_rect)
        
        # Score display
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 15, 15))
        self.screen.blit(score_text, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "health": self.player_health,
            "ammo": self.player_ammo,
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Zombie Arena")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    print("\n" + "="*40)
    print("MANUAL PLAY MODE")
    print(env.game_description)
    print(env.user_guide)
    print("="*40 + "\n")

    while not terminated:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()