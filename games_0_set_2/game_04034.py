import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    user_guide = "Controls: Use ← and → to move the catcher."

    # Must be a short, user-facing description of the game:
    game_description = "Catch falling fruit to score points. Catch 50 to win, but miss 10 and you lose!"

    # Frames auto-advance for smooth, real-time gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and game constants
        self.screen_width = 640
        self.screen_height = 400
        self.fps = 30

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 48, bold=True)
        
        # Colors
        self.COLOR_BG_TOP = (15, 25, 40)
        self.COLOR_BG_BOTTOM = (40, 60, 80)
        self.COLOR_CATCHER = (200, 150, 100)
        self.COLOR_CATCHER_RIM = (160, 110, 70)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)
        self.FRUIT_TYPES = {
            'apple': {'color': (220, 50, 50), 'points': 1, 'radius': 12},
            'orange': {'color': (240, 150, 30), 'points': 2, 'radius': 14},
            'lemon': {'color': (250, 240, 80), 'points': 5, 'radius': 10},
        }

        # Game parameters
        self.CATCHER_WIDTH = 80
        self.CATCHER_HEIGHT = 20
        self.CATCHER_SPEED = 12
        self.WIN_CONDITION = 50
        self.LOSS_CONDITION = 10
        self.MAX_STEPS = 1800 # 60 seconds at 30fps

        # Initialize state variables
        self.catcher_pos = None
        self.fruits = []
        self.particles = []
        self.score = 0
        self.fruits_caught = 0
        self.fruits_missed = 0
        self.steps = 0
        self.game_over = False
        self.game_over_message = ""
        self.spawn_timer = 0
        
        # This is called here because it needs self.np_random to be initialized
        # self.reset() is called by the parent class constructor or by the user
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.catcher_pos = pygame.Vector2(self.screen_width / 2, self.screen_height - 30)
        self.fruits = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.fruits_caught = 0
        self.fruits_missed = 0
        
        self.game_over = False
        self.game_over_message = ""
        self.spawn_timer = 0

        # Spawn initial fruits
        for _ in range(3):
            self._spawn_fruit(initial=True)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        if not self.game_over:
            # 1. Calculate reward for movement choice
            reward += self._calculate_movement_reward(movement)

            # 2. Update game state
            self._handle_input(movement)
            self._update_game_logic()
            
            # 3. Calculate rewards from game events
            reward_events = self._update_fruits_and_check_collisions()
            reward += reward_events
            
            # 4. Update particles and difficulty
            self._update_particles()
            self._update_difficulty_and_spawn()

        self.steps += 1
        
        # 5. Check termination conditions
        terminated = self._check_termination()
        if terminated and not self.game_over: # Set game over state only once
            if self.fruits_caught >= self.WIN_CONDITION:
                reward += 100
                self.game_over_message = "YOU WIN!"
            elif self.fruits_missed >= self.LOSS_CONDITION:
                reward -= 100
                self.game_over_message = "GAME OVER"
            elif self.steps >= self.MAX_STEPS:
                self.game_over_message = "TIME'S UP"
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_movement_reward(self, movement):
        if not self.fruits:
            return 0

        # Find horizontal distance to the closest fruit before moving
        catcher_x = self.catcher_pos.x
        closest_fruit = min(self.fruits, key=lambda f: abs(f['pos'].x - catcher_x))
        dist_before = abs(closest_fruit['pos'].x - catcher_x)
        
        # Simulate move
        next_catcher_x = catcher_x
        if movement == 3:  # Left
            next_catcher_x -= self.CATCHER_SPEED
        elif movement == 4:  # Right
            next_catcher_x += self.CATCHER_SPEED
        
        dist_after = abs(closest_fruit['pos'].x - next_catcher_x)

        # Reward moving towards the fruit, penalize moving away
        if dist_after < dist_before:
            return 0.1
        elif dist_after > dist_before:
            return -0.2
        return 0

    def _handle_input(self, movement):
        if movement == 3:  # Left
            self.catcher_pos.x -= self.CATCHER_SPEED
        if movement == 4:  # Right
            self.catcher_pos.x += self.CATCHER_SPEED
        
        # Clamp catcher position to screen bounds
        self.catcher_pos.x = max(self.CATCHER_WIDTH / 2, self.catcher_pos.x)
        self.catcher_pos.x = min(self.screen_width - self.CATCHER_WIDTH / 2, self.catcher_pos.x)

    def _update_game_logic(self):
        # This function can be expanded for more complex non-entity logic
        pass

    def _update_fruits_and_check_collisions(self):
        event_reward = 0
        catcher_rect = pygame.Rect(
            self.catcher_pos.x - self.CATCHER_WIDTH / 2,
            self.catcher_pos.y - self.CATCHER_HEIGHT / 2,
            self.CATCHER_WIDTH, self.CATCHER_HEIGHT
        )
        
        fruits_to_remove = []
        for fruit in self.fruits:
            # Move fruit
            fruit['pos'] += fruit['vel']
            
            # Check for catch
            fruit_rect = pygame.Rect(
                fruit['pos'].x - fruit['radius'],
                fruit['pos'].y - fruit['radius'],
                fruit['radius'] * 2, fruit['radius'] * 2
            )
            if catcher_rect.colliderect(fruit_rect):
                fruits_to_remove.append(fruit)
                self.score += fruit['points']
                self.fruits_caught += 1
                event_reward += fruit['points']
                
                # Risky catch reward
                if self.catcher_pos.x < 100 or self.catcher_pos.x > self.screen_width - 100:
                    event_reward += 5
                
                self._create_particles(fruit['pos'], fruit['color'])
                continue

            # Check for miss
            if fruit['pos'].y > self.screen_height + fruit['radius']:
                fruits_to_remove.append(fruit)
                self.fruits_missed += 1
                event_reward -= 5 # Penalty for each miss
        
        self.fruits = [f for f in self.fruits if f not in fruits_to_remove]
        return event_reward

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _update_difficulty_and_spawn(self):
        # Increase spawn rate over time
        spawn_rate_hz = 1.0 + (self.steps // 150) * 0.25 # Increase rate every 5s
        spawn_interval = self.fps / spawn_rate_hz
        
        self.spawn_timer += 1
        if self.spawn_timer >= spawn_interval:
            self.spawn_timer = 0
            self._spawn_fruit()

    def _spawn_fruit(self, initial=False):
        fruit_type_name = self.np_random.choice(list(self.FRUIT_TYPES.keys()), p=[0.6, 0.3, 0.1])
        fruit_type = self.FRUIT_TYPES[fruit_type_name]
        
        x_pos = self.np_random.integers(fruit_type['radius'], self.screen_width - fruit_type['radius'])
        y_pos = -fruit_type['radius'] if not initial else self.np_random.integers(0, self.screen_height // 2)

        # Increase fall speed over time
        base_speed = 2.0 + (self.steps // 300) * 0.5 # Increase speed every 10s
        speed_multiplier = self.np_random.uniform(0.9, 1.2)
        fall_speed = base_speed * speed_multiplier
        
        # Add horizontal drift
        drift = self.np_random.uniform(-0.5, 0.5)

        self.fruits.append({
            'pos': pygame.Vector2(x_pos, y_pos),
            'vel': pygame.Vector2(drift, fall_speed),
            'radius': fruit_type['radius'],
            'color': fruit_type['color'],
            'points': fruit_type['points']
        })

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'radius': radius
            })

    def _check_termination(self):
        return (
            self.fruits_caught >= self.WIN_CONDITION or
            self.fruits_missed >= self.LOSS_CONDITION or
            self.steps >= self.MAX_STEPS
        )
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_caught": self.fruits_caught,
            "fruits_missed": self.fruits_missed,
        }

    def _render_background(self):
        for y in range(self.screen_height):
            # Interpolate color from top to bottom
            r = self.COLOR_BG_TOP[0] + (self.COLOR_BG_BOTTOM[0] - self.COLOR_BG_TOP[0]) * y / self.screen_height
            g = self.COLOR_BG_TOP[1] + (self.COLOR_BG_BOTTOM[1] - self.COLOR_BG_TOP[1]) * y / self.screen_height
            b = self.COLOR_BG_TOP[2] + (self.COLOR_BG_BOTTOM[2] - self.COLOR_BG_TOP[2]) * y / self.screen_height
            pygame.draw.line(self.screen, (int(r), int(g), int(b)), (0, y), (self.screen_width, y))

    def _render_game(self):
        self._render_fruits()
        self._render_catcher()
        self._render_particles()

    def _render_fruits(self):
        for fruit in self.fruits:
            pos = (int(fruit['pos'].x), int(fruit['pos'].y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit['radius'], fruit['color'])

    def _render_catcher(self):
        x, y = int(self.catcher_pos.x), int(self.catcher_pos.y)
        w, h = self.CATCHER_WIDTH, self.CATCHER_HEIGHT
        r = h // 2
        
        rect = pygame.Rect(x - w / 2, y - h / 2, w, h)
        rim_rect = pygame.Rect(x - w / 2, y - h / 2, w, h-r)

        # Draw main body
        pygame.draw.rect(self.screen, self.COLOR_CATCHER, rect, border_radius=r)
        # Draw rim
        pygame.draw.rect(self.screen, self.COLOR_CATCHER_RIM, rim_rect, border_top_left_radius=r, border_top_right_radius=r)


    def _render_particles(self):
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            alpha = int(255 * (p['lifespan'] / p['max_lifespan']))
            color = (*p['color'], alpha)
            
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, int(p['radius']), int(p['radius']), int(p['radius']), color)
            self.screen.blit(temp_surf, (pos[0] - p['radius'], pos[1] - p['radius']))


    def _render_ui(self):
        # Score display
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (15, 10), self.font_ui)

        # Misses display
        misses_text = f"MISSED: {self.fruits_missed}/{self.LOSS_CONDITION}"
        miss_text_surf = self.font_ui.render(misses_text, True, self.COLOR_TEXT)
        self._draw_text(misses_text, (self.screen_width - miss_text_surf.get_width() - 15, 10), self.font_ui)

        # Game Over Message
        if self.game_over:
            self._draw_text(self.game_over_message, (self.screen_width/2, self.screen_height/2 - 50), self.font_msg, center=True)
            
            final_score_text = f"Final Score: {self.score}"
            self._draw_text(final_score_text, (self.screen_width/2, self.screen_height/2 + 10), self.font_ui, center=True)
    
    def _draw_text(self, text, pos, font, color=None, shadow_color=None, center=False):
        if color is None:
            color = self.COLOR_TEXT
        if shadow_color is None:
            shadow_color = self.COLOR_TEXT_SHADOW

        text_surf = font.render(text, True, color)
        shadow_surf = font.render(text, True, shadow_color)
        
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        
        shadow_rect = text_rect.copy()
        shadow_rect.x += 2
        shadow_rect.y += 2
        
        self.screen.blit(shadow_surf, shadow_rect)
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to visualize, or "rgb_array" for headless
    render_mode = "human" 

    if render_mode == "human":
        # For human playback, we need a real screen
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.display.set_caption("Fruit Catcher")
        real_screen = pygame.display.set_mode((640, 400))

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    done = False
    
    # Simple keyboard agent for human play
    keys_pressed = {pygame.K_LEFT: 0, pygame.K_RIGHT: 0}
    
    while not done:
        action = [0, 0, 0] # Default action: no-op

        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key in keys_pressed:
                        keys_pressed[event.key] = 1
                if event.type == pygame.KEYUP:
                    if event.key in keys_pressed:
                        keys_pressed[event.key] = 0

            if keys_pressed[pygame.K_LEFT]:
                action[0] = 3
            elif keys_pressed[pygame.K_RIGHT]:
                action[0] = 4
        else: # Random agent
             action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if render_mode == "human":
            # Blit the environment's rendered surface to the real screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.fps)
            if env.game_over:
                pygame.time.wait(2000) # Pause on game over

    env.close()
    print(f"Game Over! Final Info: {info}")