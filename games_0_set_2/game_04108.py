import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Hold [SPACE] to charge a jump. Release to leap over the red obstacles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist side-scrolling platformer. Time your jumps to traverse a "
        "procedurally generated obstacle course in your hopping space module."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GROUND_Y = 350
    
    # Colors
    COLOR_BG_TOP = (10, 20, 40)
    COLOR_BG_BOTTOM = (0, 0, 0)
    COLOR_GROUND = (100, 100, 110)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (50, 255, 150, 40)
    COLOR_OBSTACLE = (255, 80, 80)
    COLOR_OBSTACLE_GLOW = (255, 80, 80, 50)
    COLOR_PARTICLE = (200, 200, 255)
    COLOR_TEXT = (240, 240, 240)
    
    # Physics
    GRAVITY = 0.6
    WORLD_SCROLL_SPEED = 5.0
    PLAYER_X_POS = 150
    PLAYER_SIZE = 20
    
    # Jump Mechanics
    JUMP_CHARGE_RATE = 0.05
    MIN_JUMP_FORCE = 7.0
    MAX_JUMP_FORCE = 15.0
    
    # Game Rules
    WIN_DISTANCE = 1000
    MAX_STEPS = 10000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.is_on_ground = False
        self.is_charging_jump = False
        self.jump_charge = 0.0
        
        self.obstacles = []
        self.particles = []
        
        self.distance_traveled = 0.0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self.seed = None
        
        # This will be called in the first reset
        # self.reset() # Not needed as it's called by wrappers
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
            random.seed(self.seed)
        
        self.player_pos = pygame.Vector2(self.PLAYER_X_POS, self.GROUND_Y - self.PLAYER_SIZE)
        self.player_vel = pygame.Vector2(0, 0)
        
        self.is_on_ground = True
        self.is_charging_jump = False
        self.jump_charge = 0.0
        
        self.obstacles = []
        self.particles = []
        
        self.distance_traveled = 0.0
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        
        self._generate_initial_obstacles()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0.0
        terminated = False
        truncated = False
        
        # Unpack factorized action
        space_held = action[1] == 1
        
        # --- Update Game Logic ---
        self.steps += 1
        prev_distance = self.distance_traveled
        
        # Penalize for inaction while in the air (encourages planning jumps)
        if not self.is_on_ground and not self.is_charging_jump:
            reward -= 0.02
            
        # 1. Handle Input & Jump Mechanics
        self._handle_input(space_held)
        
        # 2. Update Player Physics
        self._update_player_physics()
        
        # 3. Update World (Obstacles, Particles, Distance)
        self._update_world()
        self.distance_traveled += self.WORLD_SCROLL_SPEED / 30.0 # Normalize to distance units
        
        # 4. Collision Detection
        if self._check_collisions():
            terminated = True
            self.game_over = True
            reward = -100.0
            # sfx: player_explosion
        
        # 5. Calculate Rewards
        # Distance reward
        reward += (self.distance_traveled - prev_distance) * 0.1
        
        # Obstacle clear reward
        for obs in self.obstacles:
            # obs is a dict: {'rect': pygame.Rect, 'cleared': bool}
            if not obs['cleared'] and obs['rect'].right < self.player_pos.x:
                obs['cleared'] = True
                reward += 1.0
        
        # 6. Check Termination Conditions
        if self.distance_traveled >= self.WIN_DISTANCE:
            terminated = True
            self.game_over = True
            reward += 100.0
        
        if self.steps >= self.MAX_STEPS:
            truncated = True # Use truncated for time limit
            self.game_over = True

        self.score += reward # Accumulate reward into score for info dict
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, space_held):
        # Start charging jump
        if space_held and self.is_on_ground and not self.is_charging_jump:
            self.is_charging_jump = True
            self.jump_charge = 0.0
            # sfx: charge_start
        
        # Continue charging
        if self.is_charging_jump and space_held:
            self.jump_charge = min(1.0, self.jump_charge + self.JUMP_CHARGE_RATE)

        # Release jump
        if self.is_charging_jump and not space_held:
            self.is_charging_jump = False
            self.is_on_ground = False
            jump_force = self.MIN_JUMP_FORCE + (self.jump_charge * (self.MAX_JUMP_FORCE - self.MIN_JUMP_FORCE))
            self.player_vel.y = -jump_force
            # sfx: jump_release
            self._spawn_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE / 2, self.PLAYER_SIZE), 20, 'up')

    def _update_player_physics(self):
        if not self.is_on_ground:
            self.player_vel.y += self.GRAVITY
            self.player_pos.y += self.player_vel.y
        
        # Ground collision
        if self.player_pos.y >= self.GROUND_Y - self.PLAYER_SIZE:
            self.player_pos.y = self.GROUND_Y - self.PLAYER_SIZE
            self.player_vel.y = 0
            if not self.is_on_ground:
                # Just landed
                self.is_on_ground = True
                # sfx: land
                self._spawn_particles(self.player_pos + pygame.Vector2(self.PLAYER_SIZE / 2, self.PLAYER_SIZE), 15, 'side')

    def _update_world(self):
        # Update obstacles
        new_obstacles = []
        for obs in self.obstacles:
            obs['rect'].x -= self.WORLD_SCROLL_SPEED
            if obs['rect'].right > 0:
                new_obstacles.append(obs)
        self.obstacles = new_obstacles
        
        # Spawn new obstacles
        if not self.obstacles or self.obstacles[-1]['rect'].right < self.SCREEN_WIDTH + 100:
            self._spawn_obstacle()
            
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x, self.player_pos.y, self.PLAYER_SIZE, self.PLAYER_SIZE)
        for obs in self.obstacles:
            if player_rect.colliderect(obs['rect']):
                return True
        # Check if player fell through world (shouldn't happen with ground check)
        if self.player_pos.y > self.SCREEN_HEIGHT:
            return True
        return False

    def _generate_initial_obstacles(self):
        # Start with a safe zone
        current_x = 400
        while current_x < self.SCREEN_WIDTH * 2:
            gap = random.randint(150, 250)
            current_x += gap
            
            width = random.randint(30, 60)
            height = random.randint(40, 100)
            
            self.obstacles.append({
                'rect': pygame.Rect(current_x, self.GROUND_Y - height, width, height),
                'cleared': False
            })
            current_x += width

    def _spawn_obstacle(self):
        stage = int(self.distance_traveled / 200)
        
        # Difficulty scaling
        spacing_factor = max(0.5, 1.0 - (stage * 0.02)) # Cap spacing reduction
        height_factor = min(2.5, 1.0 + (stage * 0.1)) # Cap height increase
        
        min_gap = int(100 * spacing_factor)
        max_gap = int(200 * spacing_factor)
        
        min_height = int(30 * height_factor)
        max_height = int(120 * height_factor)
        
        gap = random.randint(min_gap, max_gap)
        
        last_x = self.SCREEN_WIDTH
        if self.obstacles:
            last_x = self.obstacles[-1]['rect'].right
        
        current_x = last_x + gap
        width = random.randint(40, 80)
        height = min(self.GROUND_Y - 50, random.randint(min_height, max_height))
        
        self.obstacles.append({
            'rect': pygame.Rect(current_x, self.GROUND_Y - height, width, height),
            'cleared': False
        })
        
    def _spawn_particles(self, pos, count, direction):
        for _ in range(count):
            if direction == 'up':
                vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-3, -1))
            elif direction == 'side':
                vel = pygame.Vector2(random.uniform(-2.5, 2.5), random.uniform(-1.5, 0))
            
            self.particles.append({
                'pos': pygame.Vector2(pos), # FIX: Create a new Vector2 instance
                'vel': vel,
                'life': random.randint(20, 40),
                'radius': random.uniform(2, 5)
            })

    def _get_observation(self):
        # 1. Clear screen with background
        self._draw_background_gradient()
        
        # 2. Render all game elements
        self._render_game()
        
        # 3. Render UI overlay
        self._render_ui()
        
        # 4. Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _draw_background_gradient(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Draw Ground
        pygame.draw.line(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 3)

        # Draw Obstacles
        for obs in self.obstacles:
            # Main body
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs['rect'])
            # Glow/outline effect
            glow_rect = obs['rect'].inflate(6, 6)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, self.COLOR_OBSTACLE_GLOW, s.get_rect(), border_radius=3)
            self.screen.blit(s, glow_rect.topleft)
        
        # Draw Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40.0))
            color = (*self.COLOR_PARTICLE, alpha)
            pos = (int(p['pos'].x), int(p['pos'].y))
            radius = int(p['radius'])
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

        # Draw Player
        player_rect = pygame.Rect(int(self.player_pos.x), int(self.player_pos.y), self.PLAYER_SIZE, self.PLAYER_SIZE)
        
        # Glow
        glow_size = int(self.PLAYER_SIZE * 1.8 + self.jump_charge * 15)
        glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, self.COLOR_PLAYER_GLOW, (glow_size // 2, glow_size // 2), glow_size // 2)
        self.screen.blit(glow_surface, (player_rect.centerx - glow_size // 2, player_rect.centery - glow_size // 2))
        
        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)
        
        # Jump charge indicator
        if self.is_charging_jump:
            charge_width = int(self.PLAYER_SIZE * self.jump_charge)
            charge_rect = pygame.Rect(player_rect.left, player_rect.bottom + 5, charge_width, 4)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, charge_rect)
            
            outline_rect = pygame.Rect(player_rect.left, player_rect.bottom + 5, self.PLAYER_SIZE, 4)
            pygame.draw.rect(self.screen, self.COLOR_GROUND, outline_rect, 1)

    def _render_ui(self):
        # Distance Traveled
        dist_text = self.font_large.render(f"DIST: {int(self.distance_traveled):04d} / {self.WIN_DISTANCE}", True, self.COLOR_TEXT)
        self.screen.blit(dist_text, (10, 10))
        
        # Stage
        stage = int(self.distance_traveled / 200) + 1
        stage_text = self.font_large.render(f"STAGE: {stage}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH - stage_text.get_width() - 10, 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 40))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance": self.distance_traveled,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To use, you might need to `pip install pygame`
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Set auto_advance to False for human play to control frame rate
    env.auto_advance = False 
    
    # Action state
    action = env.action_space.sample()
    action.fill(0)

    print(env.user_guide)

    # Setup display for human play
    # This will fail if you are in a headless environment, but is useful for local testing
    try:
        pygame.display.set_caption("Hopper")
        display_surf = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    except pygame.error:
        print("\nCould not create display for human play. Running headlessly.")
        display_surf = None


    while running:
        if display_surf:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            # --- Human Controls ---
            keys = pygame.key.get_pressed()
            action[1] = 1 if keys[pygame.K_SPACE] else 0
        else: # No display, just run a random agent
            action = env.action_space.sample()

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render for human display ---
        if display_surf:
            frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            display_surf.blit(frame_surface, (0, 0))
            pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Distance: {info['distance']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0.0
            action.fill(0)
            if display_surf:
                pygame.time.wait(2000) # Pause before restarting
            else: # In headless mode, just exit after one episode
                running = False

        env.clock.tick(30) # Control FPS for human play

    env.close()