
# Generated: 2025-08-28T00:49:38.085939
# Source Brief: brief_03910.md
# Brief Index: 3910

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


# Helper class for the player
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.width = 20
        self.height = 20
        self.color = (0, 255, 100)
        self.on_platform = True
        self.jump_charge = 0
        self.max_jump_charge = 100
        self.breathing_timer = 0

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, screen):
        # Breathing effect when on platform
        size_offset = 0
        if self.on_platform:
            size_offset = math.sin(self.breathing_timer) * 2
        
        rect = pygame.Rect(
            int(self.x - size_offset / 2), 
            int(self.y - size_offset / 2), 
            int(self.width + size_offset), 
            int(self.height + size_offset)
        )
        pygame.draw.rect(screen, self.color, rect, border_radius=3)
        pygame.draw.rect(screen, (200, 255, 200), rect, width=2, border_radius=3)

# Helper class for platforms
class Platform:
    def __init__(self, x, y, width, height, is_goal=False, is_start=False):
        self.base_x = x
        self.base_y = y
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x, y, width, height)
        self.color = (200, 200, 255) if not is_goal else (255, 215, 0)
        self.outline_color = (255, 255, 255) if not is_goal else (255, 255, 100)
        self.is_goal = is_goal
        self.is_start = is_start

        # Sinusoidal motion parameters
        if not is_start and not is_goal:
            self.amplitude = random.uniform(5, 20)
            self.frequency = random.uniform(0.01, 0.03)
            self.phase = random.uniform(0, 2 * math.pi)
        else:
            self.amplitude = 0
            self.frequency = 0
            self.phase = 0

    def update(self, time_step, speed_multiplier):
        self.rect.y = self.base_y + self.amplitude * math.sin(self.frequency * speed_multiplier * time_step + self.phase)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=4)
        pygame.draw.rect(screen, self.outline_color, self.rect, width=2, border_radius=4)

# Helper class for collectibles
class Collectible:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 6
        self.color = (255, 255, 0)
        self.glow_timer = random.uniform(0, 2 * math.pi)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    def update(self):
        self.glow_timer += 0.1

    def draw(self, screen):
        # Pulsing glow effect
        glow_radius = self.radius + abs(math.sin(self.glow_timer) * 2)
        glow_alpha = 100 + math.sin(self.glow_timer) * 50
        
        # Using pygame.gfxdraw for anti-aliased circles
        pygame.gfxdraw.aacircle(screen, int(self.x), int(self.y), int(glow_radius), (*self.color, int(glow_alpha)))
        pygame.gfxdraw.filled_circle(screen, int(self.x), int(self.y), int(glow_radius), (*self.color, int(glow_alpha)))
        
        pygame.gfxdraw.aacircle(screen, int(self.x), int(self.y), self.radius, self.color)
        pygame.gfxdraw.filled_circle(screen, int(self.x), int(self.y), self.radius, self.color)

# Helper class for particles
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-3, 1)
        self.life = random.randint(15, 30)
        self.color = color
        self.size = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1 # particle gravity
        self.life -= 1

    def draw(self, screen):
        if self.life > 0:
            pygame.draw.rect(screen, self.color, (int(self.x), int(self.y), self.size, self.size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→ to move on platforms. Hold Space to charge a jump, release to leap. "
        "Directional keys at the moment of release influence jump direction."
    )

    game_description = (
        "Hop between procedurally generated platforms to reach the goal at the top. "
        "Collect yellow orbs for points and try to finish as fast as possible."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width = 640
        self.height = 400
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.large_font = pygame.font.Font(None, 50)

        # Game constants
        self.GRAVITY = 0.4
        self.PLAYER_X_SPEED = 3.0
        self.AIR_CONTROL_X = 0.3
        self.JUMP_POWER_BASE = 8.0
        self.JUMP_POWER_SCALING = 0.07
        self.MAX_EPISODE_STEPS = 2000

        # Colors
        self.COLOR_BG_TOP = (10, 0, 30)
        self.COLOR_BG_BOTTOM = (0, 0, 0)
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self._generate_platforms()
        self._generate_collectibles()

        start_platform = self.platforms[0]
        self.player = Player(start_platform.rect.centerx, start_platform.rect.top - 20)
        self.start_y = self.player.y
        self.last_platform_y = self.player.y
        self.prev_space_held = False
        self.platform_speed_multiplier = 1.0

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def _generate_platforms(self):
        self.platforms = []
        
        # Start platform
        start_plat = Platform(self.width / 2 - 50, self.height - 50, 100, 15, is_start=True)
        self.platforms.append(start_plat)
        
        current_y = start_plat.rect.y
        
        # Generate intermediate platforms
        while current_y > 100:
            last_plat = self.platforms[-1]
            
            w = random.uniform(60, 120)
            h = 15
            
            # Ensure next platform is reachable
            max_jump_dist_x = 120
            x = last_plat.rect.centerx + random.uniform(-max_jump_dist_x, max_jump_dist_x)
            x = np.clip(x, w/2, self.width - w/2) # Keep in bounds
            
            max_jump_dist_y = 130
            min_jump_dist_y = 40
            y = last_plat.rect.y - random.uniform(min_jump_dist_y, max_jump_dist_y)
            
            self.platforms.append(Platform(x - w/2, y, w, h))
            current_y = y

        # Goal platform
        goal_plat = Platform(0, 40, self.width, 20, is_goal=True)
        self.platforms.append(goal_plat)

    def _generate_collectibles(self):
        self.collectibles = []
        # Place collectibles on some intermediate platforms
        for plat in self.platforms[1:-1]:
            if random.random() < 0.4: # 40% chance of having a collectible
                self.collectibles.append(Collectible(plat.rect.centerx, plat.rect.top - 15))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Update game logic ---
        self.steps += 1
        
        # Update platform speed
        if self.steps % 100 == 0 and self.steps > 0:
            self.platform_speed_multiplier += 0.01

        # Update platforms, collectibles, particles
        for plat in self.platforms:
            plat.update(self.steps, self.platform_speed_multiplier)
        for collectible in self.collectibles:
            collectible.update()
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.life > 0]
        
        # --- Handle player input and physics ---
        
        # Horizontal movement
        if self.player.on_platform:
            self.player.vx = 0
            if movement == 3: # Left
                self.player.vx = -self.PLAYER_X_SPEED
            elif movement == 4: # Right
                self.player.vx = self.PLAYER_X_SPEED
        else: # Air control
            if movement == 3:
                self.player.vx -= self.AIR_CONTROL_X
            elif movement == 4:
                self.player.vx += self.AIR_CONTROL_X
            self.player.vx = np.clip(self.player.vx, -self.PLAYER_X_SPEED, self.PLAYER_X_SPEED)

        # Jumping
        if self.player.on_platform:
            self.player.breathing_timer += 0.05
            if space_held:
                self.player.jump_charge = min(self.player.max_jump_charge, self.player.jump_charge + 2)
            
            # Jump on release
            if not space_held and self.prev_space_held:
                jump_power = self.JUMP_POWER_BASE + (self.player.jump_charge / self.player.max_jump_charge) * self.JUMP_POWER_SCALING * 100
                self.player.vy = -jump_power
                
                # Add horizontal component based on movement keys
                if movement == 3:
                    self.player.vx = -self.PLAYER_X_SPEED * 1.5
                elif movement == 4:
                    self.player.vx = self.PLAYER_X_SPEED * 1.5
                else:
                    self.player.vx = 0 # Straight up jump
                
                self.player.on_platform = None # Set to None to indicate just jumped
                self.player.jump_charge = 0
                # Sound effect: Jump
                self._create_particles(self.player.x + self.player.width/2, self.player.y + self.player.height, self.player.color, 15)

        self.prev_space_held = space_held
        
        # Apply physics
        if not self.player.on_platform:
            self.player.vy += self.GRAVITY
        
        self.player.x += self.player.vx
        self.player.y += self.player.vy
        
        # --- Collision Detection ---
        
        # Player-Platform collision
        if self.player.vy > 0 and self.player.on_platform is None: # Just jumped, ignore collision for a moment
             if self.player.vy > -self.JUMP_POWER_BASE / 2: # Allow passing through platform from below
                 self.player.on_platform = False

        if self.player.vy > 0 and not self.player.on_platform:
            player_rect = self.player.get_rect()
            for plat in self.platforms:
                if player_rect.colliderect(plat.rect) and player_rect.bottom < plat.rect.centery:
                    self.player.y = plat.rect.top - self.player.height
                    self.player.vy = 0
                    self.player.on_platform = plat
                    self.player.breathing_timer = 0
                    # Sound effect: Land
                    self._create_particles(self.player.x + self.player.width/2, plat.rect.top, (200,200,255), 10)
                    
                    if plat.rect.y > self.last_platform_y:
                        reward -= 1
                    self.last_platform_y = plat.rect.y
                    
                    if plat.is_goal:
                        self.game_won = True
                    break

        # Player-Collectible collision
        player_rect = self.player.get_rect()
        for collectible in self.collectibles[:]:
            if player_rect.colliderect(collectible.get_rect()):
                self.collectibles.remove(collectible)
                reward += 5
                self.score += 50 # Display score
                # Sound effect: Collect
                self._create_particles(collectible.x, collectible.y, collectible.color, 20)

        # Boundary checks
        if self.player.x < 0:
            self.player.x = 0
            self.player.vx = 0
        if self.player.x > self.width - self.player.width:
            self.player.x = self.width - self.player.width
            self.player.vx = 0

        # --- Calculate Reward & Termination ---
        reward += 0.1 if self.player.y < self.start_y else -0.1
        self.score = max(0, self.score + reward) # Update score for display

        terminated = False
        if self.player.y > self.height:
            terminated = True
            reward = -50
            self.game_over = True
        elif self.game_won:
            terminated = True
            reward = 100
            self.game_over = True
        elif self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def _get_observation(self):
        # Draw background gradient
        for y in range(self.height):
            ratio = y / self.height
            color = (
                int(self.COLOR_BG_TOP[0] * (1 - ratio) + self.COLOR_BG_BOTTOM[0] * ratio),
                int(self.COLOR_BG_TOP[1] * (1 - ratio) + self.COLOR_BG_BOTTOM[1] * ratio),
                int(self.COLOR_BG_TOP[2] * (1 - ratio) + self.COLOR_BG_BOTTOM[2] * ratio)
            )
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))

        # Render all game elements
        for plat in self.platforms:
            plat.draw(self.screen)
        for collectible in self.collectibles:
            collectible.draw(self.screen)
        for p in self.particles:
            p.draw(self.screen)
        
        self.player.draw(self.screen)
        
        # Render UI overlay
        self._render_ui()

        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Score
        score_text = self.font.render(f"SCORE: {int(self.score)}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        # Time
        time_text = self.font.render(f"TIME: {self.steps}", True, (255, 255, 255))
        time_rect = time_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(time_text, time_rect)

        # Jump charge bar
        if self.player.jump_charge > 0:
            charge_ratio = self.player.jump_charge / self.player.max_jump_charge
            bar_width = 40
            bar_height = 8
            bar_x = self.player.x + self.player.width / 2 - bar_width / 2
            bar_y = self.player.y - 15
            
            # Interpolate color from white to red
            color = (255, 255 - charge_ratio * 255, 255 - charge_ratio * 255)
            
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width * charge_ratio, bar_height))
            pygame.draw.rect(self.screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)

    def _render_game_over(self):
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "YOU REACHED THE TOP!" if self.game_won else "GAME OVER"
        text = self.large_font.render(message, True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.width / 2, self.height / 2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "game_won": self.game_won
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    import os
    # To run headless for checks, uncomment the line below
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    # --- To run with a window for visualization ---
    try:
        env = GameEnv(render_mode="rgb_array")
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Arcade Hopper")
        clock = pygame.time.Clock()
        running = True
        env.reset()

        while running:
            movement, space, shift = 0, 0, 0
            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1

            action = [movement, space, shift]
            obs, reward, terminated, truncated, info = env.step(action)

            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Score: {info['score']}, Steps: {info['steps']}")
                pygame.time.wait(2000)
                env.reset()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            clock.tick(30) # Match the auto-advance rate
    finally:
        env.close()