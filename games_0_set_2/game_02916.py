
# Generated: 2025-08-27T21:48:18.987303
# Source Brief: brief_02916.md
# Brief Index: 2916

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to shoot. Survive the horde!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive hordes of procedurally generated zombies in a top-down arena shooter. "
        "Clear waves to score points, but watch your health!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 50)
        
        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_WALL = (60, 60, 70)
        self.COLOR_PLAYER = (0, 255, 127) # Spring Green
        self.COLOR_ZOMBIE = (220, 20, 60) # Crimson
        self.COLOR_PROJECTILE = (255, 255, 0) # Yellow
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BAR = (46, 204, 113)
        self.COLOR_HEALTH_BAR_BG = (120, 40, 40)

        # Game constants
        self.PLAYER_SPEED = 4.0
        self.PLAYER_RADIUS = 10
        self.PLAYER_MAX_HEALTH = 100
        self.PROJECTILE_SPEED = 8.0
        self.PROJECTILE_RADIUS = 3
        self.PROJECTILE_DAMAGE = 5
        self.SHOOT_COOLDOWN_MAX = 5 # 6 shots per second at 30fps
        self.ZOMBIE_RADIUS = 8
        self.ZOMBIE_BASE_HEALTH = 10
        self.ZOMBIE_CONTACT_DAMAGE = 1
        self.MAX_STEPS = 5000
        
        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_angle = None
        self.last_move_dir = None
        self.zombies = None
        self.projectiles = None
        self.particles = None
        self.wave = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.shoot_cooldown = None
        self.zombie_speed = None
        
        # This will be called once to set up the initial state
        self.reset()

        # Run validation check
        # self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_angle = -math.pi / 2  # Start facing up
        self.last_move_dir = pygame.Vector2(0, -1)
        
        self.zombies = []
        self.projectiles = []
        self.particles = []
        
        self.wave = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.shoot_cooldown = 0
        self.zombie_speed = 1.0

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Handle Input and Cooldowns ---
        self._handle_input(action)
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1

        # --- Update Game Logic ---
        self._update_player()
        reward += self._update_projectiles()
        reward += self._update_zombies()
        self._update_particles()
        
        # --- Check for Wave Completion ---
        if not self.zombies:
            reward += 10 * self.wave
            self.score += 100 * self.wave
            self.wave += 1
            self._spawn_wave()
            # Sound: wave_complete.wav

        # --- Check Termination Conditions ---
        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        if self.player_health <= 0 and not self.game_over:
            reward = -100 # Large penalty for dying
            self.game_over = True
            # Sound: player_death.wav
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        move_dir = pygame.Vector2(0, 0)
        if movement == 1:  # Up
            move_dir.y = -1
        elif movement == 2:  # Down
            move_dir.y = 1
        elif movement == 3:  # Left
            move_dir.x = -1
        elif movement == 4:  # Right
            move_dir.x = 1

        if move_dir.length() > 0:
            self.player_pos += move_dir.normalize() * self.PLAYER_SPEED
            self.last_move_dir = move_dir.copy()
            self.player_angle = self.last_move_dir.angle_to(pygame.Vector2(1, 0)) * -math.pi / 180.0
        
        if space_held and self.shoot_cooldown == 0:
            self._fire_projectile()
            self.shoot_cooldown = self.SHOOT_COOLDOWN_MAX

    def _fire_projectile(self):
        # Sound: shoot.wav
        start_pos = self.player_pos + self.last_move_dir.normalize() * (self.PLAYER_RADIUS + 5)
        self.projectiles.append({
            "pos": start_pos,
            "vel": self.last_move_dir.normalize() * self.PROJECTILE_SPEED
        })
        self._create_muzzle_flash(start_pos, self.last_move_dir)

    def _update_player(self):
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)
        
    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            
            is_alive = True
            # Check off-screen
            if not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                reward -= 0.01 # Penalty for missing
                is_alive = False

            # Check collision with zombies
            for z in self.zombies:
                if p["pos"].distance_to(z["pos"]) < self.PROJECTILE_RADIUS + self.ZOMBIE_RADIUS:
                    z["health"] -= self.PROJECTILE_DAMAGE
                    reward += 0.1 # Reward for hitting
                    is_alive = False
                    # Sound: hit_zombie.wav
                    break
            
            if is_alive:
                projectiles_to_keep.append(p)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_zombies(self):
        reward = 0
        self.zombie_speed = min(3.0, 1.0 + self.steps / 500 * 0.1)
        
        zombies_to_keep = []
        for z in self.zombies:
            if z["health"] <= 0:
                reward += 1.0 # Reward for kill
                self.score += 10
                self._create_explosion(z["pos"])
                # Sound: zombie_die.wav
                continue

            # Move towards player
            direction = (self.player_pos - z["pos"]).normalize() if self.player_pos != z["pos"] else pygame.Vector2(0,0)
            z["pos"] += direction * self.zombie_speed
            
            # Check collision with player
            if z["pos"].distance_to(self.player_pos) < self.ZOMBIE_RADIUS + self.PLAYER_RADIUS:
                self.player_health -= self.ZOMBIE_CONTACT_DAMAGE
                self.player_health = max(0, self.player_health)
                # Sound: player_hurt.wav
                # No direct reward penalty, handled by episode termination penalty

            zombies_to_keep.append(z)

        self.zombies = zombies_to_keep
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifetime"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            p["radius"] *= 0.95

    def _spawn_wave(self):
        num_zombies = 3 + 2 * self.wave
        for _ in range(num_zombies):
            # Spawn zombies at the edges of the screen
            side = self.np_random.integers(4)
            if side == 0: # top
                x, y = self.np_random.uniform(0, self.WIDTH), -self.ZOMBIE_RADIUS
            elif side == 1: # bottom
                x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ZOMBIE_RADIUS
            elif side == 2: # left
                x, y = -self.ZOMBIE_RADIUS, self.np_random.uniform(0, self.HEIGHT)
            else: # right
                x, y = self.WIDTH + self.ZOMBIE_RADIUS, self.np_random.uniform(0, self.HEIGHT)
            
            self.zombies.append({
                "pos": pygame.Vector2(x, y),
                "health": self.ZOMBIE_BASE_HEALTH
            })
    
    def _create_explosion(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            color = random.choice([(255, 69, 0), (255, 140, 0), (255, 165, 0)])
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(2, 6),
                "lifetime": self.np_random.integers(15, 30),
                "color": color
            })

    def _create_muzzle_flash(self, pos, direction):
        for _ in range(5):
            # Create a cone of particles
            angle_offset = self.np_random.uniform(-0.5, 0.5)
            angle = math.atan2(direction.y, direction.x) + angle_offset
            speed = self.np_random.uniform(2, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": self.np_random.uniform(1, 3),
                "lifetime": self.np_random.integers(3, 7),
                "color": (255, 255, 224) # Light Yellow
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles (drawn first, so they are behind other objects)
        for p in self.particles:
            alpha = int(255 * (p["lifetime"] / 30.0))
            if p["radius"] > 1:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), (*p["color"], alpha)
                )

        # Render zombies
        for z in self.zombies:
            pygame.draw.circle(self.screen, self.COLOR_ZOMBIE, (int(z["pos"].x), int(z["pos"].y)), self.ZOMBIE_RADIUS)
            pygame.draw.circle(self.screen, tuple(c*0.7 for c in self.COLOR_ZOMBIE), (int(z["pos"].x), int(z["pos"].y)), self.ZOMBIE_RADIUS, 2)

        # Render projectiles
        for p in self.projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE, (int(p["pos"].x), int(p["pos"].y)), self.PROJECTILE_RADIUS)
            
        # Render player
        if self.player_health > 0:
            # Body
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, (int(self.player_pos.x), int(self.player_pos.y)), self.PLAYER_RADIUS)
            # Direction indicator (a small triangle)
            p1 = self.player_pos + self.last_move_dir.normalize() * self.PLAYER_RADIUS
            p2 = self.player_pos + self.last_move_dir.normalize().rotate(135) * self.PLAYER_RADIUS * 0.6
            p3 = self.player_pos + self.last_move_dir.normalize().rotate(-135) * self.PLAYER_RADIUS * 0.6
            pygame.draw.polygon(self.screen, tuple(c*0.8 for c in self.COLOR_PLAYER), [(int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), (int(p3.x), int(p3.y))])
    
    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), bar_height))
        
        # Wave Text
        wave_text = self.font_small.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Score Text
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 30))

        if self.game_over:
            game_over_text = self.font_large.render("GAME OVER", True, self.COLOR_ZOMBIE)
            text_rect = game_over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(game_over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "zombies_left": len(self.zombies),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set this to "dummy" for headless execution, or remove for visual debugging
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    env.validate_implementation()
    
    # --- For visual debugging ---
    # To visually debug, comment out the os.environ line above and uncomment the following block.
    # You will need to have pygame installed and a display available.

    # os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "quartz" depending on your OS
    # env = GameEnv(render_mode="rgb_array")
    # obs, info = env.reset()
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # pygame.display.set_caption("Zombie Arena")
    # running = True
    # total_reward = 0
    #
    # while running:
    #     action = [0, 0, 0] # Default no-op
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]:
    #         action[0] = 1
    #     elif keys[pygame.K_DOWN]:
    #         action[0] = 2
    #     elif keys[pygame.K_LEFT]:
    #         action[0] = 3
    #     elif keys[pygame.K_RIGHT]:
    #         action[0] = 4
    #
    #     if keys[pygame.K_SPACE]:
    #         action[1] = 1
    #
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_reward += reward
    #
    #     # Convert observation back to a Pygame surface for display
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
    #         pygame.time.wait(2000)
    #         obs, info = env.reset()
    #         total_reward = 0
    #
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #
    #     env.clock.tick(30) # Limit to 30 FPS
    #
    # env.close()