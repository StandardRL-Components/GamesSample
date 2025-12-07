
# Generated: 2025-08-27T21:56:31.189990
# Source Brief: brief_02958.md
# Brief Index: 2958

        
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
        "Controls: ↑/↓ to aim. Press SPACE to jump. Press SHIFT to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a jumping robot to blast waves of progressively faster enemies in a side-scrolling arena."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # For auto_advance=True, this is the target framerate
        self.MAX_STEPS = 1500 # Increased for more gameplay time

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GROUND = (60, 60, 80)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_GUN = (200, 255, 220)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_PROJECTILE = (80, 200, 255)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_BG = (120, 40, 40)
        self.COLOR_HEALTH_FG = (40, 200, 40)
        self.COLOR_PARTICLE_HIT = (255, 255, 100)
        self.COLOR_PARTICLE_DEATH = self.COLOR_ENEMY

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # Game state variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.enemy_base_speed = 0
        self.enemy_kill_count = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.rng = None

        # Call reset to initialize the state
        self.reset()
        
        # This check is for development and ensures API compliance.
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            # Fallback if no seed is provided
            if self.rng is None:
                 self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.enemy_kill_count = 0
        self.last_space_held = False
        self.last_shift_held = False

        # Player state
        self.player = {
            "pos": pygame.Vector2(100, self.HEIGHT - 50),
            "size": pygame.Vector2(24, 32),
            "velocity": pygame.Vector2(0, 0),
            "on_ground": True,
            "health": 100,
            "aim_angle": 0,
            "shoot_cooldown": 0,
            "squash": 1.0, # For animation
        }

        # Clear lists
        self.enemies.clear()
        self.projectiles.clear()
        self.particles.clear()

        # Enemy setup
        self.enemy_base_speed = 1.0
        num_enemies = 7
        for i in range(num_enemies):
            self.enemies.append({
                "pos": pygame.Vector2(
                    self.WIDTH + 100 + i * 80,
                    self.rng.integers(100, self.HEIGHT - 50 - 20)
                ),
                "size": pygame.Vector2(20, 20),
                "health": 10,
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing and just return the final state.
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = -0.01 # Small penalty for existing, encourages speed

        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Aiming
        if movement == 1: # Up
            self.player["aim_angle"] = max(-math.pi / 4, self.player["aim_angle"] - 0.1)
        elif movement == 2: # Down
            self.player["aim_angle"] = min(math.pi / 4, self.player["aim_angle"] + 0.1)

        # Jumping (on key press, not hold)
        if space_held and not self.last_space_held and self.player["on_ground"]:
            self.player["velocity"].y = -11 # JUMP_STRENGTH
            self.player["on_ground"] = False
            self.player["squash"] = 1.5 # Stretch effect
            # sfx: jump

        # Shooting (on key press, not hold)
        if shift_held and not self.last_shift_held and self.player["shoot_cooldown"] <= 0:
            self._fire_projectile()
            # sfx: shoot

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update Game State ---
        self._update_player()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # --- Collisions and Game Logic ---
        reward += self._handle_collisions()

        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.player["health"] <= 0:
            terminated = True
            # No specific reward for losing
        elif not self.enemies:
            terminated = True
            reward += 50 # Big reward for winning
            self.score += 5000 # Bonus score for winning
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _fire_projectile(self):
        self.player["shoot_cooldown"] = 10 # 1/3 second cooldown at 30fps
        gun_offset = pygame.Vector2(self.player["size"].x / 2, 0).rotate(-math.degrees(self.player["aim_angle"]))
        start_pos = self.player["pos"] + pygame.Vector2(self.player["size"].x/2, self.player["size"].y/2) + gun_offset
        
        velocity = pygame.Vector2(15, 0).rotate(-math.degrees(self.player["aim_angle"]))
        
        self.projectiles.append({
            "pos": start_pos,
            "velocity": velocity,
            "radius": 4,
        })
        # Muzzle flash particle
        for _ in range(5):
            p_vel = pygame.Vector2(self.rng.random() * 2, 0).rotate(-math.degrees(self.player["aim_angle"]) + self.rng.integers(-30, 30))
            self.particles.append({
                "pos": start_pos.copy(), "vel": p_vel, "radius": self.rng.integers(2, 5),
                "color": self.COLOR_PARTICLE_HIT, "lifespan": 5
            })

    def _update_player(self):
        # Physics
        GRAVITY = 0.6
        self.player["velocity"].y += GRAVITY
        self.player["pos"] += self.player["velocity"]
        
        # Animation
        self.player["squash"] = max(1.0, self.player["squash"] * 0.9)

        # Ground collision
        ground_y = self.HEIGHT - 50
        if self.player["pos"].y + self.player["size"].y >= ground_y:
            if not self.player["on_ground"]:
                self.player["squash"] = 0.7 # Squash effect on landing
            self.player["pos"].y = ground_y - self.player["size"].y
            self.player["velocity"].y = 0
            self.player["on_ground"] = True

        # Cooldown
        if self.player["shoot_cooldown"] > 0:
            self.player["shoot_cooldown"] -= 1

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"] += p["velocity"]
            if not (0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT):
                self.projectiles.remove(p)

    def _update_enemies(self):
        current_speed = self.enemy_base_speed + self.enemy_kill_count * 0.1
        for e in self.enemies:
            e["pos"].x -= current_speed

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["radius"] -= 0.2
            p["lifespan"] -= 1
            if p["radius"] <= 0 or p["lifespan"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player["pos"], self.player["size"])

        # Projectiles vs Enemies
        for p in self.projectiles[:]:
            p_rect = pygame.Rect(p["pos"].x - p["radius"], p["pos"].y - p["radius"], p["radius"]*2, p["radius"]*2)
            for e in self.enemies[:]:
                e_rect = pygame.Rect(e["pos"], e["size"])
                if e_rect.colliderect(p_rect):
                    e["health"] -= 10
                    reward += 0.1
                    self.score += 10
                    
                    # Hit particles
                    for _ in range(3):
                        self.particles.append({
                            "pos": p["pos"].copy(), "vel": pygame.Vector2(self.rng.uniform(-1, 1), self.rng.uniform(-1, 1)),
                            "radius": self.rng.integers(2, 4), "color": self.COLOR_PARTICLE_HIT, "lifespan": 10
                        })
                    
                    if p in self.projectiles: self.projectiles.remove(p) # sfx: hit
                    
                    if e["health"] <= 0:
                        reward += 1.0
                        self.score += 100
                        self.enemy_kill_count += 1
                        
                        # Death particles
                        for _ in range(20):
                            self.particles.append({
                                "pos": e["pos"] + e["size"]/2, "vel": pygame.Vector2(self.rng.uniform(-3, 3), self.rng.uniform(-3, 3)),
                                "radius": self.rng.integers(2, 6), "color": self.COLOR_PARTICLE_DEATH, "lifespan": 20
                            })
                        
                        self.enemies.remove(e) # sfx: explosion
                    break

        # Enemies vs Player
        for e in self.enemies[:]:
            e_rect = pygame.Rect(e["pos"], e["size"])
            if e_rect.colliderect(player_rect):
                self.player["health"] -= 10
                self.enemies.remove(e) # Enemy sacrifices itself
                # sfx: player_damage
                # Damage particles
                for _ in range(10):
                    self.particles.append({
                        "pos": self.player["pos"] + self.player["size"]/2, "vel": pygame.Vector2(self.rng.uniform(-2, 2), self.rng.uniform(-2, 2)),
                        "radius": self.rng.integers(2, 5), "color": self.COLOR_PLAYER, "lifespan": 15
                    })
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.HEIGHT - 50, self.WIDTH, 50))

        # Draw particles
        for p in self.particles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], max(0, int(p["radius"])), p["color"])

        # Draw enemies
        for e in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (e["pos"], e["size"]))

        # Draw projectiles
        for p in self.projectiles:
            pos = (int(p["pos"].x), int(p["pos"].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p["radius"]), self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(p["radius"]), self.COLOR_PROJECTILE)

        # Draw player
        squashed_size = pygame.Vector2(self.player["size"].x / self.player["squash"], self.player["size"].y * self.player["squash"])
        squashed_pos = self.player["pos"] + (self.player["size"] - squashed_size) / 2
        player_rect = pygame.Rect(squashed_pos, squashed_size)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=3)
        
        # Draw gun
        gun_length = 20
        gun_thickness = 8
        gun_center = player_rect.center
        gun_end = pygame.Vector2(gun_length, 0).rotate(-math.degrees(self.player["aim_angle"]))
        
        p1 = pygame.Vector2(0, -gun_thickness/2).rotate(-math.degrees(self.player["aim_angle"])) + gun_center
        p2 = pygame.Vector2(0, gun_thickness/2).rotate(-math.degrees(self.player["aim_angle"])) + gun_center
        p3 = p2 + gun_end
        p4 = p1 + gun_end
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3, p4), self.COLOR_PLAYER_GUN)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3, p4), self.COLOR_PLAYER_GUN)


    def _render_ui(self):
        # Health bar
        health_pct = max(0, self.player["health"] / 100.0)
        health_bar_rect_bg = pygame.Rect(10, 10, 200, 20)
        health_bar_rect_fg = pygame.Rect(10, 10, 200 * health_pct, 20)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, health_bar_rect_bg, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, health_bar_rect_fg, border_radius=3)

        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, self.HEIGHT - 30))

        # Enemy count
        enemy_text = self.font_small.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_TEXT)
        self.screen.blit(enemy_text, (self.WIDTH - enemy_text.get_width() - 10, 10))

        # Game Over / Win Text
        if self.game_over:
            if not self.enemies:
                win_text = self.font_large.render("VICTORY!", True, self.COLOR_PLAYER)
                text_rect = win_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
                self.screen.blit(win_text, text_rect)
            else:
                lose_text = self.font_large.render("GAME OVER", True, self.COLOR_ENEMY)
                text_rect = lose_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
                self.screen.blit(lose_text, text_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "enemies_remaining": len(self.enemies),
        }

    def close(self):
        pygame.font.quit()
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
if __name__ == "__main__":
    import os
    # Set a dummy video driver to run pygame headlessly
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    
    # --- To play manually ---
    # This requires a graphical backend. Comment out the dummy video driver line above.
    # from gymnasium.utils.play import play
    # play(env, fps=30, keys_to_action={
    #     # Movement is action[0]
    #     ("w",): [1, 0, 0],
    #     ("s",): [2, 0, 0],
    #     # Space is action[1]
    #     (" ",): [0, 1, 0],
    #     # Shift is action[2]
    #     (pygame.K_LSHIFT,): [0, 0, 1],
    #     # Combinations
    #     ("w", " "): [1, 1, 0],
    #     ("s", " "): [2, 1, 0],
    #     ("w", pygame.K_LSHIFT): [1, 0, 1],
    #     ("s", pygame.K_LSHIFT): [2, 0, 1],
    #     (" ", pygame.K_LSHIFT): [0, 1, 1],
    #     ("w", " ", pygame.K_LSHIFT): [1, 1, 1],
    #     ("s", " ", pygame.K_LSHIFT): [2, 1, 1],
    # })

    # --- To test the environment with random actions ---
    print("Testing with random actions...")
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    step_count = 0
    
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if terminated or truncated:
            break
            
    print(f"Episode finished after {step_count} steps.")
    print(f"Final Info: {info}")
    print(f"Total Reward: {total_reward}")
    
    env.close()