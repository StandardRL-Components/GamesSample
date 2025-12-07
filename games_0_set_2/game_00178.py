
# Generated: 2025-08-27T12:50:36.855967
# Source Brief: brief_00178.md
# Brief Index: 178

        
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
    """
    A 2D side-scrolling platformer where the player navigates a procedurally
    generated forest, jumping over obstacles and collecting coins.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: ←→ to run, ↑ or Space to jump. Collect coins and reach the end!"
    )
    game_description = (
        "A fast-paced 2D platformer. Navigate a procedural forest, collect coins, "
        "and leap over perilous pits to reach the goal."
    )

    # Frame advance setting
    auto_advance = True

    # --- Constants ---
    # Screen
    WIDTH, HEIGHT = 640, 400
    # Colors
    COLOR_SKY = (135, 206, 235)
    COLOR_GROUND = (34, 139, 34)
    COLOR_PIT = (20, 10, 0)
    COLOR_PLAYER = (255, 69, 0)
    COLOR_PLAYER_GLOW = (255, 140, 0, 100)
    COLOR_COIN = (255, 215, 0)
    COLOR_PLATFORM = (139, 69, 19)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (50, 50, 50)
    COLOR_GOAL = (255, 255, 255)
    # Physics
    GRAVITY = 0.3
    PLAYER_SPEED = 4.0
    JUMP_STRENGTH = -8.0
    FRICTION = 0.9
    # Level Generation
    LEVEL_LENGTH = 3000  # Total length of the level in pixels
    TILE_SIZE = 40
    GROUND_Y = HEIGHT - 50
    # Game Rules
    MAX_STEPS = 2500
    CHECKPOINT_INTERVAL = 400 # pixels

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.ui_font = pygame.font.Font(None, 36)
        self.end_font = pygame.font.Font(None, 72)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_size = pygame.Vector2(24, 32)
        self.is_grounded = False
        self.coyote_time = 0

        self.camera_x = 0.0
        self.platforms = []
        self.coins = []
        self.pits = []
        self.particles = []
        self.background_trees = []

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_checkpoint = 0
        self.pit_frequency = 0.0
        self.platform_move_speed = 0.0
        self.end_goal_rect = pygame.Rect(0,0,0,0)
        self.rng = np.random.default_rng()

        self.reset()
        # self.validate_implementation() # Optional self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_checkpoint = 0

        self.player_pos = pygame.Vector2(100, self.GROUND_Y - self.player_size.y)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_grounded = False
        self.coyote_time = 0

        self.camera_x = 0.0
        self.pit_frequency = 0.01
        self.platform_move_speed = 0.0

        self._generate_level()
        self.particles.clear()

        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.platforms.clear()
        self.coins.clear()
        self.pits.clear()
        self.background_trees.clear()

        # Generate a safe starting area
        for i in range(5):
             self.platforms.append(pygame.Rect(i * self.TILE_SIZE, self.GROUND_Y, self.TILE_SIZE, self.HEIGHT - self.GROUND_Y))

        x = 5 * self.TILE_SIZE
        last_feature_x = x
        while x < self.LEVEL_LENGTH:
            # Ensure there's a minimum gap between features
            if x - last_feature_x < self.TILE_SIZE * 2:
                self.platforms.append(pygame.Rect(x, self.GROUND_Y, self.TILE_SIZE, self.HEIGHT - self.GROUND_Y))
                x += self.TILE_SIZE
                continue

            rand = self.rng.random()
            if rand < self.pit_frequency: # Create a pit
                pit_width = self.rng.integers(2, 5) * self.TILE_SIZE
                self.pits.append(pygame.Rect(x, self.GROUND_Y, pit_width, self.HEIGHT - self.GROUND_Y))
                x += pit_width
                last_feature_x = x
            elif rand < self.pit_frequency + 0.15: # Create a platform group
                plat_y = self.GROUND_Y - self.rng.integers(2, 5) * self.TILE_SIZE
                plat_length = self.rng.integers(3, 7)
                move_type = 0 # 0=static, 1=vertical, 2=horizontal
                if self.rng.random() < 0.3: # Chance for moving platform
                    move_type = self.rng.choice([1, 2])

                for i in range(plat_length):
                    platform_rect = pygame.Rect(x + i * self.TILE_SIZE, plat_y, self.TILE_SIZE, 20)
                    self.platforms.append({
                        "rect": platform_rect,
                        "move_type": move_type,
                        "origin_y": plat_y,
                        "origin_x": x + i * self.TILE_SIZE,
                        "move_range": self.rng.integers(40, 80),
                        "move_speed": self.platform_move_speed,
                    })
                    if self.rng.random() < 0.5: # Place a coin
                        self.coins.append(pygame.Rect(x + i * self.TILE_SIZE + 10, plat_y - 30, 20, 20))
                x += plat_length * self.TILE_SIZE
                last_feature_x = x
            else: # Create ground
                self.platforms.append(pygame.Rect(x, self.GROUND_Y, self.TILE_SIZE, self.HEIGHT - self.GROUND_Y))
                if self.rng.random() < 0.1: # Place a coin on the ground
                    self.coins.append(pygame.Rect(x + 10, self.GROUND_Y - 30, 20, 20))
                x += self.TILE_SIZE

        self.end_goal_rect = pygame.Rect(self.LEVEL_LENGTH + 100, self.GROUND_Y - 100, 20, 100)
        self.platforms.append(pygame.Rect(self.LEVEL_LENGTH, self.GROUND_Y, 200, self.HEIGHT-self.GROUND_Y))

        # Generate background trees
        for _ in range(100):
            self.background_trees.append({
                "x": self.rng.integers(0, self.LEVEL_LENGTH * 2),
                "y": self.GROUND_Y,
                "height": self.rng.integers(50, 150),
                "parallax": self.rng.uniform(0.2, 0.6)
            })
        self.background_trees.sort(key=lambda t: t["parallax"])


    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Time penalty

        # --- 1. Handle Input ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED

        if (movement == 1 or space_held) and (self.is_grounded or self.coyote_time > 0):
            self.player_vel.y = self.JUMP_STRENGTH
            self.is_grounded = False
            self.coyote_time = 0
            # sfx: jump
            self._create_particles(self.player_pos + pygame.Vector2(self.player_size.x / 2, self.player_size.y), count=5, color=(200,200,200))


        # --- 2. Update Physics & Game State ---
        self.player_vel.y += self.GRAVITY
        self.player_vel.x *= self.FRICTION
        self.player_pos += self.player_vel

        # Update dynamic platforms
        for plat in self.platforms:
            if isinstance(plat, dict) and plat["move_type"] != 0:
                phase = math.sin(self.steps * 0.02 * plat["move_speed"])
                if plat["move_type"] == 1: # Vertical
                    plat["rect"].y = plat["origin_y"] + phase * plat["move_range"]
                elif plat["move_type"] == 2: # Horizontal
                    plat["rect"].x = plat["origin_x"] + phase * plat["move_range"]

        # Update particles
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

        # Update difficulty
        if self.steps > 0 and self.steps % 100 == 0:
            self.pit_frequency = min(0.2, self.pit_frequency + 0.001)
        if self.steps > 0 and self.steps % 200 == 0:
            self.platform_move_speed = min(2.0, self.platform_move_speed + 0.05)


        # --- 3. Collision Detection ---
        player_rect = pygame.Rect(self.player_pos, self.player_size)
        
        # Grounded check
        on_any_surface = False
        if player_rect.bottom >= self.GROUND_Y:
            # Check if we are over a pit
            in_pit = False
            for pit in self.pits:
                if pit.left < player_rect.centerx < pit.right:
                    in_pit = True
                    break
            if not in_pit:
                self.player_pos.y = self.GROUND_Y - self.player_size.y
                self.player_vel.y = 0
                if not self.is_grounded: # Landing
                    self._create_particles(player_rect.midbottom, count=8, color=(100, 200, 100))
                    # sfx: land
                on_any_surface = True

        # Platform collision
        for plat_obj in self.platforms:
            plat_rect = plat_obj["rect"] if isinstance(plat_obj, dict) else plat_obj
            if player_rect.colliderect(plat_rect) and self.player_vel.y > 0:
                # Check if player was above the platform in the previous frame
                if player_rect.bottom - self.player_vel.y <= plat_rect.top:
                    self.player_pos.y = plat_rect.top - self.player_size.y
                    self.player_vel.y = 0
                    if not self.is_grounded: # Landing
                        self._create_particles(player_rect.midbottom, count=8, color=(160, 82, 45))
                        # sfx: land
                    on_any_surface = True
                    break # Only land on one platform
        
        self.is_grounded = on_any_surface
        if self.is_grounded:
            self.coyote_time = 5 # frames of coyote time
        else:
            self.coyote_time -= 1


        # --- 4. Check Interactions & Termination ---
        # Coin collection
        collected_indices = player_rect.collidelistall(self.coins)
        if collected_indices:
            for i in sorted(collected_indices, reverse=True):
                self._create_particles(self.coins[i].center, count=10, color=self.COLOR_COIN)
                del self.coins[i]
                self.score += 1
                reward += 1
                # sfx: coin_collect
        
        # Checkpoints
        current_checkpoint = int(self.player_pos.x // self.CHECKPOINT_INTERVAL)
        if current_checkpoint > self.last_checkpoint:
            self.last_checkpoint = current_checkpoint
            reward += 5
            # sfx: checkpoint
            self._create_particles(player_rect.center, count=20, color=(100, 100, 255))


        # Termination conditions
        terminated = False
        # Fell off world
        if player_rect.top > self.HEIGHT:
            reward -= 50
            terminated = True
            # sfx: fall
        # Reached goal
        if player_rect.colliderect(self.end_goal_rect):
            reward += 50
            terminated = True
            # sfx: win
        # Max steps
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        # Update camera to follow player smoothly
        target_camera_x = self.player_pos.x - self.WIDTH / 3
        self.camera_x += (target_camera_x - self.camera_x) * 0.1

        # --- Render everything ---
        # Background
        self.screen.fill(self.COLOR_SKY)

        # Parallax background trees
        for tree in self.background_trees:
            tree_x = tree["x"] - self.camera_x * tree["parallax"]
            if -50 < tree_x < self.WIDTH + 50:
                 pygame.draw.rect(self.screen, (101, 67, 33, 150), (tree_x, tree["y"] - tree["height"], 10, tree["height"])) # Trunk
                 pygame.draw.circle(self.screen, (34, 139, 34, 150), (tree_x + 5, tree["y"] - tree["height"]), 30) # Leaves


        # Game elements (offset by camera)
        # Pits
        for pit in self.pits:
            pygame.draw.rect(self.screen, self.COLOR_PIT, pit.move(-self.camera_x, 0))
        # Platforms & Ground
        for plat_obj in self.platforms:
            plat_rect = plat_obj["rect"] if isinstance(plat_obj, dict) else plat_obj
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, plat_rect.move(-self.camera_x, 0))
        # Coins
        for coin in self.coins:
            # Bobbing animation
            bob_offset = math.sin(self.steps * 0.1 + coin.x) * 3
            r = coin.move(-self.camera_x, bob_offset)
            pygame.gfxdraw.filled_circle(self.screen, r.centerx, r.centery, int(r.width/2), self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, r.centerx, r.centery, int(r.width/2), (255,255,255))
        # Goal
        pygame.draw.rect(self.screen, self.COLOR_GOAL, self.end_goal_rect.move(-self.camera_x, 0))
        
        # Player
        player_rect_on_screen = pygame.Rect(self.player_pos.x - self.camera_x, self.player_pos.y, self.player_size.x, self.player_size.y)
        # Squash and stretch based on vertical velocity
        squash = min(5, abs(self.player_vel.y))
        stretch = max(0, self.player_vel.y * 1.2)
        player_render_rect = player_rect_on_screen.inflate(squash, -stretch)
        player_render_rect.midbottom = player_rect_on_screen.midbottom
        
        # Glow effect for player
        glow_surf = pygame.Surface(player_render_rect.size, pygame.SRCALPHA)
        glow_rect = glow_surf.get_rect()
        pygame.draw.ellipse(glow_surf, self.COLOR_PLAYER_GLOW, glow_rect)
        self.screen.blit(glow_surf, player_render_rect.topleft)

        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_render_rect, border_radius=4)

        # Particles
        for p in self.particles:
            pos = (int(p["pos"].x - self.camera_x), int(p["pos"].y))
            radius = int(p["lifespan"] / p["max_life"] * 5)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, p["color"])

        # --- UI Overlay ---
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_ui(self):
        # Score Text
        score_text = f"Coins: {self.score}"
        self._draw_text(score_text, (10, 10), self.ui_font)

        # Progress Bar
        progress = self.player_pos.x / self.LEVEL_LENGTH
        bar_width = self.WIDTH - 20
        bar_height = 15
        bar_x, bar_y = 10, self.HEIGHT - bar_height - 10
        pygame.draw.rect(self.screen, self.COLOR_TEXT_SHADOW, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_SKY, (bar_x+2, bar_y+2, bar_width-4, bar_height-4), border_radius=4)
        fill_width = max(0, min(bar_width - 4, (bar_width - 4) * progress))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (bar_x+2, bar_y+2, fill_width, bar_height-4), border_radius=4)

        # Game Over Text
        if self.game_over:
            msg = "GOAL!" if self.player_pos.x >= self.LEVEL_LENGTH else "GAME OVER"
            self._draw_text(msg, self.screen.get_rect().center, self.end_font, center=True)


    def _draw_text(self, text, pos, font, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW, center=False):
        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        self.screen.blit(shadow_surf, text_rect.move(2, 2))
        self.screen.blit(text_surf, text_rect)

    def _create_particles(self, position, count=10, color=(255,255,255)):
        for _ in range(count):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            lifespan = self.rng.integers(15, 40)
            self.particles.append({
                "pos": pygame.Vector2(position),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "lifespan": lifespan,
                "max_life": lifespan,
                "color": color
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_x": self.player_pos.x,
            "player_y": self.player_pos.y,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Platformer Game")
    human_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    total_reward = 0.0
    
    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_UP]:
            movement = 1
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to screen ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0.0
                terminated = False

        env.clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()