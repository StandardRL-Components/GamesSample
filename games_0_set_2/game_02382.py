
# Generated: 2025-08-27T20:12:27.719570
# Source Brief: brief_02382.md
# Brief Index: 2382

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot. Hold Shift to dash."
    )

    # User-facing description of the game
    game_description = (
        "Survive for 60 seconds against an onslaught of zombies in a top-down arena shooter. Use your agility and firepower to stay alive."
    )

    # Frames auto-advance for smooth, real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME_SECONDS = 60
        self.MAX_STEPS = self.MAX_TIME_SECONDS * self.FPS

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_PLAYER = (0, 255, 150)
        self.COLOR_PLAYER_GLOW = (0, 255, 150, 50)
        self.COLOR_ZOMBIE = (255, 50, 50)
        self.COLOR_ZOMBIE_TOP = (200, 40, 40)
        self.COLOR_PROJECTILE = (255, 255, 0)
        self.COLOR_BLOOD = (180, 0, 0)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_HEALTH_BAR = (0, 200, 100)
        self.COLOR_HEALTH_BAR_BG = (80, 80, 80)
        
        # Game parameters
        self.PLAYER_SPEED = 4
        self.PLAYER_HEALTH_MAX = 100
        self.ZOMBIE_SPEED = 1.2
        self.ZOMBIE_HEALTH = 10
        self.ZOMBIE_DAMAGE = 20
        self.ZOMBIE_COUNT_INITIAL = 20
        self.PROJECTILE_SPEED = 15
        self.PROJECTILE_DAMAGE = 10
        self.SHOOT_COOLDOWN = 5  # frames
        self.DASH_SPEED = 18
        self.DASH_DURATION = 4 # frames
        self.DASH_COOLDOWN = 45 # frames
        self.PROXIMITY_THRESHOLD = 75 # pixels for proximity penalty

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_score = pygame.font.Font(None, 36)

        # --- State Variables ---
        self.player_pos = None
        self.player_health = None
        self.player_facing_dir = None
        self.player_last_movement_dir = None
        self.shoot_cooldown_timer = None
        self.dash_cooldown_timer = None
        self.dash_active_timer = None
        self.zombies = None
        self.projectiles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_space_held = False

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_facing_dir = pygame.Vector2(0, -1) # Start facing up
        self.player_last_movement_dir = pygame.Vector2(0, -1)
        
        self.shoot_cooldown_timer = 0
        self.dash_cooldown_timer = 0
        self.dash_active_timer = 0

        self.zombies = self._spawn_zombies(self.ZOMBIE_COUNT_INITIAL)
        self.projectiles = []
        self.particles = []
        self.last_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = self.game_over

        if not terminated:
            self._handle_input(action)
            
            reward += self._update_player()
            reward += self._update_projectiles()
            reward += self._update_zombies()
            self._update_particles()
            
            self.steps += 1
            
            # Check for termination conditions
            if self.player_health <= 0:
                terminated = True
                reward -= 100 # Penalty for dying
                # sfx: player_death
            elif self.steps >= self.MAX_STEPS:
                terminated = True
                reward += 100 # Reward for surviving
                # sfx: level_win

        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement_action, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        self.move_vector = pygame.Vector2(0, 0)
        if movement_action == 1: self.move_vector.y = -1 # Up
        elif movement_action == 2: self.move_vector.y = 1  # Down
        elif movement_action == 3: self.move_vector.x = -1 # Left
        elif movement_action == 4: self.move_vector.x = 1  # Right
        
        if self.move_vector.length() > 0:
            self.move_vector.normalize_ip()
            self.player_last_movement_dir = self.move_vector.copy()
        
        self.player_facing_dir = self.player_last_movement_dir

        # --- Dashing ---
        if shift_held and self.dash_cooldown_timer == 0 and self.dash_active_timer == 0:
            self.dash_active_timer = self.DASH_DURATION
            self.dash_cooldown_timer = self.DASH_COOLDOWN
            # sfx: player_dash

        # --- Shooting ---
        # Fire on press, not hold, to avoid single action firing multiple shots
        if space_held and not self.last_space_held and self.shoot_cooldown_timer == 0:
            self._fire_projectile()
        self.last_space_held = space_held

    def _fire_projectile(self):
        self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
        # Spawn projectile slightly in front of the player
        start_pos = self.player_pos + self.player_facing_dir * 20
        self.projectiles.append({
            "pos": start_pos,
            "vel": self.player_facing_dir * self.PROJECTILE_SPEED
        })
        # sfx: player_shoot
        # Muzzle flash effect
        for _ in range(10):
            vel = self.player_facing_dir * random.uniform(2, 5) + pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3))
            self.particles.append({
                "pos": start_pos.copy(),
                "vel": vel,
                "radius": random.uniform(1, 4),
                "lifespan": random.randint(3, 8),
                "color": self.COLOR_PROJECTILE
            })

    def _update_player(self):
        # Update cooldowns
        if self.shoot_cooldown_timer > 0: self.shoot_cooldown_timer -= 1
        if self.dash_cooldown_timer > 0: self.dash_cooldown_timer -= 1
        
        # Determine speed
        if self.dash_active_timer > 0:
            speed = self.DASH_SPEED
            self.dash_active_timer -= 1
            # Dash trail effect
            self.particles.append({
                "pos": self.player_pos.copy(),
                "vel": pygame.Vector2(0,0),
                "radius": 10,
                "lifespan": 5,
                "color": self.COLOR_PLAYER_GLOW
            })
        else:
            speed = self.PLAYER_SPEED

        # Update position
        self.player_pos += self.move_vector * speed
        
        # Clamp position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)
        
        # Proximity penalty
        reward = 0
        for zombie in self.zombies:
            if self.player_pos.distance_to(zombie["pos"]) < self.PROXIMITY_THRESHOLD:
                reward -= 0.01
        return reward

    def _update_projectiles(self):
        reward = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p["pos"] += p["vel"]
            
            hit_zombie = False
            for z in self.zombies:
                if p["pos"].distance_to(z["pos"]) < 12: # Collision radius
                    z["health"] -= self.PROJECTILE_DAMAGE
                    reward += 0.1 # Reward for hitting
                    hit_zombie = True
                    # sfx: zombie_hit
                    self._create_blood_splatter(p["pos"])
                    break
            
            # Keep projectile if it's within bounds and hasn't hit anything
            if not hit_zombie and 0 < p["pos"].x < self.WIDTH and 0 < p["pos"].y < self.HEIGHT:
                projectiles_to_keep.append(p)
        
        self.projectiles = projectiles_to_keep
        return reward

    def _update_zombies(self):
        reward = 0
        zombies_to_keep = []
        for z in self.zombies:
            # Movement
            direction = (self.player_pos - z["pos"])
            if direction.length() > 0:
                direction.normalize_ip()
            z["pos"] += direction * self.ZOMBIE_SPEED

            # Player collision
            if z["pos"].distance_to(self.player_pos) < 15 and self.dash_active_timer == 0:
                self.player_health = max(0, self.player_health - self.ZOMBIE_DAMAGE)
                # sfx: player_hurt
                # Knockback player slightly
                if direction.length() > 0:
                    self.player_pos -= direction * 10 

            # Check if alive
            if z["health"] > 0:
                zombies_to_keep.append(z)
            else:
                reward += 1 # Reward for killing
                # sfx: zombie_die
                self._create_blood_splatter(z["pos"], num_particles=30, big_splatter=True)
        
        self.zombies = zombies_to_keep
        return reward

    def _update_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            if p["lifespan"] > 0:
                particles_to_keep.append(p)
        self.particles = particles_to_keep

    def _spawn_zombies(self, count):
        zombies = []
        for _ in range(count):
            # Spawn on the perimeter
            edge = self.np_random.integers(4)
            if edge == 0: # Top
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), -20)
            elif edge == 1: # Bottom
                pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.HEIGHT + 20)
            elif edge == 2: # Left
                pos = pygame.Vector2(-20, self.np_random.uniform(0, self.HEIGHT))
            else: # Right
                pos = pygame.Vector2(self.WIDTH + 20, self.np_random.uniform(0, self.HEIGHT))
            
            zombies.append({
                "pos": pos,
                "health": self.ZOMBIE_HEALTH
            })
        return zombies

    def _create_blood_splatter(self, pos, num_particles=15, big_splatter=False):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4) if not big_splatter else random.uniform(2, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "radius": random.uniform(1, 3) if not big_splatter else random.uniform(2, 5),
                "lifespan": random.randint(15, 40),
                "color": self.COLOR_BLOOD
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render particles (drawn first, to be in the background)
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 40))
            color = p["color"]
            if len(p["color"]) == 4: # Handle RGBA colors
                color = (*p["color"][:3], int(p["color"][3] * (p["lifespan"] / 10)))
            
            if len(color) == 4:
                # Special handling for semi-transparent particles like dash trail
                surf = pygame.Surface((p["radius"] * 2, p["radius"] * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (p["radius"], p["radius"]), p["radius"])
                self.screen.blit(surf, (int(p["pos"].x - p["radius"]), int(p["pos"].y - p["radius"])))
            else:
                 pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"].x), int(p["pos"].y), int(p["radius"]), (*color, alpha)
                )

        # Render zombies
        for z in self.zombies:
            x, y = int(z["pos"].x), int(z["pos"].y)
            s = 10 # half-size
            # Draw as isometric cube
            points_base = [(x-s, y), (x, y+s/2), (x+s, y), (x, y-s/2)]
            points_top = [(x, y-s/2), (x+s, y), (x, y+s/2), (x-s, y-s)]
            # This isometric cube logic is a bit complex, let's simplify to a diamond shape
            # which gives a good pseudo-isometric feel
            iso_poly = [(x, y - s), (x + s, y), (x, y + s), (x - s, y)]
            top_poly = [(x, y-s), (x+s, y), (x, y), (x-s,y)]
            pygame.gfxdraw.aapolygon(self.screen, iso_poly, self.COLOR_ZOMBIE)
            pygame.gfxdraw.filled_polygon(self.screen, iso_poly, self.COLOR_ZOMBIE)
            pygame.gfxdraw.filled_polygon(self.screen, [(x-s,y),(x,y-s),(x+s,y),(x,y)], self.COLOR_ZOMBIE_TOP)


        # Render projectiles
        for p in self.projectiles:
            start = p["pos"] - p["vel"].normalize() * 10
            end = p["pos"]
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, (int(start.x), int(start.y)), (int(end.x), int(end.y)), 3)

        # Render player
        if self.player_health > 0:
            p_pos = self.player_pos
            angle = self.player_facing_dir.angle_to(pygame.Vector2(0, -1))
            size = 12
            
            # Glow effect
            glow_radius = size * 2
            if self.dash_active_timer > 0: glow_radius *= 1.5
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(s, (int(p_pos.x - glow_radius), int(p_pos.y - glow_radius)))
            
            # Rotated triangle
            points = [
                pygame.Vector2(0, -size),
                pygame.Vector2(-size * 0.8, size * 0.8),
                pygame.Vector2(size * 0.8, size * 0.8),
            ]
            rotated_points = [p.rotate(-angle) + p_pos for p in points]
            int_points = [(int(p.x), int(p.y)) for p in rotated_points]
            
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Health bar
        health_ratio = self.player_health / self.PLAYER_HEALTH_MAX
        bar_width = 200
        bar_height = 15
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (10, 10, int(bar_width * health_ratio), bar_height))

        # Timer
        time_left = self.MAX_TIME_SECONDS - (self.steps / self.FPS)
        time_text = f"TIME: {max(0, int(time_left)):02d}"
        text_surf = self.font_ui.render(time_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 10, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        text_surf = self.font_score.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(text_surf, (self.WIDTH // 2 - text_surf.get_width() // 2, self.HEIGHT - 40))

    def _get_info(self):
        # Update score based on rewards (this is a common pattern for gym envs)
        # The reward from step() is for the RL agent, self.score is for the human player
        # For simplicity, we can just use the same value.
        # Let's define score as number of zombies killed.
        current_score = (self.ZOMBIE_COUNT_INITIAL - len(self.zombies))
        self.score = current_score
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "time_left": self.MAX_TIME_SECONDS - (self.steps / self.FPS)
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # Set the video driver to dummy to run headless
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    
    # To play manually
    # Requires a window, so comment out the dummy driver line above
    # And change render_mode in __init__ to "human" if you were to implement it
    # For this brief, we only need rgb_array, so we'll simulate manual play.
    
    pygame.display.set_caption("Zombie Survival")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop for manual play
    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        mov = 0 # none
        if keys[pygame.K_UP]: mov = 1
        elif keys[pygame.K_DOWN]: mov = 2
        elif keys[pygame.K_LEFT]: mov = 3
        elif keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Game Over! Final Info: {info}")
    env.close()