
# Generated: 2025-08-28T04:46:20.587336
# Source Brief: brief_02415.md
# Brief Index: 2415

        
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
        "Controls: Use arrow keys to select a dish. Press space to serve the selected dish."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive the night shift at the Midnight Diner by serving customers while avoiding cursed dishes."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 5000
        self.MAX_LIVES = 3
        self.CUSTOMER_ARRIVAL_INTERVAL = 60 # Steps between customers
        self.CURSE_CHANCE = 0.25
        self.FEEDBACK_DURATION = 45 # Frames for feedback text to show

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_feedback = pygame.font.Font(None, 48)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_COUNTER_TOP = (60, 45, 40)
        self.COLOR_COUNTER_FRONT = (45, 30, 25)
        self.COLOR_FLOOR = (40, 45, 60)
        self.COLOR_TEXT = (230, 230, 240)
        self.COLOR_HEART = (220, 50, 50)
        self.COLOR_CURSED = (255, 60, 120)
        self.COLOR_GOOD = (60, 255, 120)
        self.COLOR_SELECTOR = (255, 220, 50)
        self.COLOR_SPEECH_BUBBLE = (200, 200, 210)
        self.COLOR_SPEECH_BUBBLE_OUTLINE = (10, 10, 20)

        # Dish visual definitions
        self.DISH_TYPES = {
            0: {'name': 'Burger', 'color1': (150, 90, 50), 'color2': (255, 215, 0)},
            1: {'name': 'Fries', 'color1': (210, 40, 40), 'color2': (255, 230, 100)},
            2: {'name': 'Soda', 'color1': (180, 200, 220), 'color2': (250, 80, 80)},
            3: {'name': 'Pie', 'color1': (200, 160, 100), 'color2': (180, 50, 50)}
        }
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.game_over = False
        self.customer = None
        self.dishes = []
        self.selected_dish_idx = -1
        self.last_serve_feedback = None
        self.particles = []
        self.screen_shake = 0
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.MAX_LIVES
        self.game_over = False
        self.customer = None
        self.dishes = []
        self.selected_dish_idx = -1
        self.last_serve_feedback = None
        self.particles = []
        self.screen_shake = 0
        
        self._spawn_customer()
        self._generate_dishes()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_pressed = action[1] == 1
        
        self.steps += 1
        reward = 0
        
        self._update_particles()
        if self.screen_shake > 0:
            self.screen_shake -= 1

        if self.last_serve_feedback and self.last_serve_feedback['timer'] > 0:
            self.last_serve_feedback['timer'] -= 1

        # Handle dish selection
        if movement == 1: self.selected_dish_idx = 0 # Up -> Top-Left
        elif movement == 4: self.selected_dish_idx = 1 # Right -> Top-Right
        elif movement == 3: self.selected_dish_idx = 2 # Left -> Bottom-Left
        elif movement == 2: self.selected_dish_idx = 3 # Down -> Bottom-Right
        
        # Handle serving action
        if space_pressed and self.selected_dish_idx != -1 and self.customer is not None:
            served_dish = self.dishes[self.selected_dish_idx]
            
            # Only serving the correct type of dish has an effect
            if served_dish['id'] == self.customer['dish_request']:
                if served_dish['is_cursed']:
                    # Served a cursed dish
                    self.lives -= 1
                    reward = -5
                    self.last_serve_feedback = {'text': 'CURSED!', 'color': self.COLOR_CURSED, 'timer': self.FEEDBACK_DURATION}
                    self.screen_shake = 20
                    self._create_particles(served_dish['pos'], 30, self.COLOR_CURSED)
                    # Sound: Cursed sfx
                else:
                    # Served a correct, normal dish
                    self.score += 1
                    reward = 1
                    self.last_serve_feedback = {'text': 'THANKS!', 'color': self.COLOR_GOOD, 'timer': self.FEEDBACK_DURATION}
                    self._create_particles(served_dish['pos'], 20, self.COLOR_GOOD)
                    # Sound: Correct sfx
                
                # Customer leaves, reset for next
                self.customer = None
                self.dishes = []
                self.selected_dish_idx = -1

        # Spawn new customer if table is empty
        if self.customer is None and not self.game_over:
            if self.steps % self.CUSTOMER_ARRIVAL_INTERVAL == 0 or self.steps == 1:
                self._spawn_customer()
                self._generate_dishes()

        # Check for termination conditions
        terminated = self.lives <= 0 or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.lives > 0 and self.steps >= self.MAX_STEPS:
                # Survival victory
                reward += 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _spawn_customer(self):
        self.customer = {
            'dish_request': self.np_random.integers(0, len(self.DISH_TYPES)),
            'pos': (self.SCREEN_WIDTH // 2, 80)
        }

    def _generate_dishes(self):
        self.dishes = []
        if self.customer is None: return

        dish_ids = list(self.DISH_TYPES.keys())
        # Ensure the requested dish is one of the options
        if self.customer['dish_request'] in dish_ids:
            dish_ids.remove(self.customer['dish_request'])
        
        other_dishes = self.np_random.choice(dish_ids, 3, replace=False).tolist()
        options = [self.customer['dish_request']] + other_dishes
        self.np_random.shuffle(options)
        
        positions = [
            (self.SCREEN_WIDTH // 2 - 80, self.SCREEN_HEIGHT // 2 + 20), # Top-left
            (self.SCREEN_WIDTH // 2 + 80, self.SCREEN_HEIGHT // 2 + 20), # Top-right
            (self.SCREEN_WIDTH // 2 - 80, self.SCREEN_HEIGHT // 2 + 100), # Bottom-left
            (self.SCREEN_WIDTH // 2 + 80, self.SCREEN_HEIGHT // 2 + 100) # Bottom-right
        ]
        
        for i in range(4):
            dish_id = options[i]
            is_cursed = (dish_id == self.customer['dish_request']) and (self.np_random.random() < self.CURSE_CHANCE)
            self.dishes.append({
                'id': dish_id,
                'is_cursed': is_cursed,
                'pos': positions[i]
            })

    def _get_observation(self):
        # Create a temporary surface for screen shake
        render_surface = self.screen.copy()
        render_surface.fill(self.COLOR_BG)
        
        self._render_game(render_surface)
        self._render_ui(render_surface)

        if self.screen_shake > 0:
            offset_x = self.np_random.integers(-8, 9)
            offset_y = self.np_random.integers(-8, 9)
            self.screen.fill(self.COLOR_BG)
            self.screen.blit(render_surface, (offset_x, offset_y))
        else:
            self.screen.blit(render_surface, (0, 0))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self, surface):
        # Draw floor and counter
        floor_poly = [(0, 180), (self.SCREEN_WIDTH, 180), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), (0, self.SCREEN_HEIGHT)]
        pygame.gfxdraw.filled_polygon(surface, floor_poly, self.COLOR_FLOOR)
        
        counter_top_poly = [(0, 250), (self.SCREEN_WIDTH, 250), (self.SCREEN_WIDTH, 320), (0, 320)]
        pygame.gfxdraw.filled_polygon(surface, counter_top_poly, self.COLOR_COUNTER_TOP)
        
        counter_front_poly = [(0, 320), (self.SCREEN_WIDTH, 320), (self.SCREEN_WIDTH, 350), (0, 350)]
        pygame.gfxdraw.filled_polygon(surface, counter_front_poly, self.COLOR_COUNTER_FRONT)
        pygame.draw.line(surface, (0,0,0), (0, 320), (self.SCREEN_WIDTH, 320), 2)

        # Draw customer and speech bubble
        if self.customer:
            self._render_customer(surface, self.customer)
            self._render_speech_bubble(surface, self.customer)

        # Draw dishes
        for i, dish in enumerate(self.dishes):
            self._render_dish(surface, dish, dish['pos'])
            if i == self.selected_dish_idx:
                self._render_selector(surface, dish['pos'])

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(surface, p['color'], p['pos'], int(p['size']))

    def _render_customer(self, surface, customer):
        x, y = customer['pos']
        # Simple silhouette
        pygame.draw.circle(surface, (10, 10, 15), (x, y - 10), 25) # Head
        body_poly = [(x-30, y+15), (x+30, y+15), (x+15, y+60), (x-15, y+60)]
        pygame.gfxdraw.filled_polygon(surface, body_poly, (10, 10, 15))

    def _render_speech_bubble(self, surface, customer):
        bubble_rect = pygame.Rect(0, 0, 80, 60)
        bubble_rect.center = (customer['pos'][0] + 90, customer['pos'][1] - 30)
        
        pygame.draw.rect(surface, self.COLOR_SPEECH_BUBBLE, bubble_rect, border_radius=10)
        pygame.draw.rect(surface, self.COLOR_SPEECH_BUBBLE_OUTLINE, bubble_rect, 2, border_radius=10)
        
        # Pointer
        p1 = (bubble_rect.left, bubble_rect.centery)
        p2 = (bubble_rect.left - 15, bubble_rect.centery - 10)
        p3 = (bubble_rect.left, bubble_rect.centery - 20)
        pygame.draw.polygon(surface, self.COLOR_SPEECH_BUBBLE, [p1,p2,p3])
        pygame.draw.line(surface, self.COLOR_SPEECH_BUBBLE_OUTLINE, p1, p2, 2)
        pygame.draw.line(surface, self.COLOR_SPEECH_BUBBLE_OUTLINE, p2, p3, 2)

        # Dish inside bubble
        dish_in_bubble = {'id': customer['dish_request']}
        self._render_dish(surface, dish_in_bubble, bubble_rect.center, scale=0.6)

    def _render_dish(self, surface, dish, pos, scale=1.0):
        dish_info = self.DISH_TYPES[dish['id']]
        x, y = pos
        
        if dish_info['name'] == 'Burger':
            pygame.draw.rect(surface, dish_info['color1'], (x - 20*scale, y - 10*scale, 40*scale, 10*scale), border_radius=int(3*scale))
            pygame.draw.rect(surface, dish_info['color2'], (x - 20*scale, y - 2*scale, 40*scale, 4*scale))
            pygame.draw.rect(surface, dish_info['color1'], (x - 20*scale, y + 0*scale, 40*scale, 10*scale), border_radius=int(3*scale))
        elif dish_info['name'] == 'Fries':
            pygame.draw.rect(surface, dish_info['color1'], (x - 15*scale, y - 5*scale, 30*scale, 25*scale))
            for i in range(5):
                pygame.draw.line(surface, dish_info['color2'], (x - 12*scale + i*6*scale, y - 5*scale), (x - 12*scale + i*6*scale, y - 20*scale), int(5*scale))
        elif dish_info['name'] == 'Soda':
            pygame.draw.polygon(surface, dish_info['color1'], [(x-15*scale, y+15*scale), (x+15*scale, y+15*scale), (x+10*scale, y-15*scale), (x-10*scale, y-15*scale)])
            pygame.draw.line(surface, dish_info['color2'], (x+5*scale, y-15*scale), (x+15*scale, y-25*scale), int(4*scale))
        elif dish_info['name'] == 'Pie':
            pygame.draw.polygon(surface, dish_info['color1'], [(x, y - 15*scale), (x - 20*scale, y + 10*scale), (x + 20*scale, y + 10*scale)])
            pygame.draw.circle(surface, dish_info['color2'], (x, y), int(8*scale))

    def _render_selector(self, surface, pos):
        x, y = pos
        alpha = 100 + (math.sin(self.steps * 0.2) * 50 + 50)
        radius = 40 + (math.sin(self.steps * 0.2) * 5)
        
        temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        color = self.COLOR_SELECTOR + (int(alpha),)
        pygame.gfxdraw.filled_circle(temp_surf, int(radius), int(radius), int(radius), color)
        surface.blit(temp_surf, (int(x - radius), int(y - radius)))

    def _render_ui(self, surface):
        # Score
        score_text = self.font_ui.render(f"SERVED: {self.score}", True, self.COLOR_TEXT)
        surface.blit(score_text, (10, 10))
        
        # Lives (Hearts)
        for i in range(self.MAX_LIVES):
            pos = (self.SCREEN_WIDTH - 30 - i * 35, 15)
            color = self.COLOR_HEART if i < self.lives else (50, 50, 60)
            pygame.draw.circle(surface, color, (pos[0] + 7, pos[1] + 7), 7)
            pygame.draw.circle(surface, color, (pos[0] - 7, pos[1] + 7), 7)
            pygame.draw.polygon(surface, color, [(pos[0]-14, pos[1]+7), (pos[0]+14, pos[1]+7), (pos[0], pos[1]+22)])

        # Timer
        time_left = self.MAX_STEPS - self.steps
        hours = max(0, 5 - int(time_left / (self.MAX_STEPS / 6)))
        minutes = max(0, 59 - int((time_left % (self.MAX_STEPS / 6)) / (self.MAX_STEPS / 360)))
        time_str = f"{hours:02d}:{minutes:02d} AM"
        time_text = self.font_ui.render(time_str, True, self.COLOR_TEXT)
        surface.blit(time_text, (10, 40))

        # Feedback text
        if self.last_serve_feedback and self.last_serve_feedback['timer'] > 0:
            alpha = min(255, self.last_serve_feedback['timer'] * 20)
            feedback_surf = self.font_feedback.render(self.last_serve_feedback['text'], True, self.last_serve_feedback['color'])
            feedback_surf.set_alpha(alpha)
            pos = feedback_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 50))
            surface.blit(feedback_surf, pos)

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.lives <= 0:
                end_text_str = "YOU'RE FIRED"
                color = self.COLOR_CURSED
            else:
                end_text_str = "SHIFT COMPLETE"
                color = self.COLOR_GOOD

            end_text = self.font_game_over.render(end_text_str, True, color)
            end_pos = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            
            surface.blit(overlay, (0,0))
            surface.blit(end_text, end_pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
        }

    def _create_particles(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'size': self.np_random.random() * 4 + 2,
                'life': self.np_random.integers(20, 40)
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] *= 0.95
    
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Midnight Diner")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Mapping from Pygame keys to action[0] values
    key_to_movement = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0 # Unused in this game
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        for key, move_val in key_to_movement.items():
            if keys[key]:
                movement_action = move_val
                break # Take first pressed key
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Limit to 30 FPS
        
    env.close()